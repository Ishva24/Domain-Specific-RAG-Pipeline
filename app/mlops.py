"""
MLflow Integration — Tracing, Prompt Registry, and RAGAS Evaluation
====================================================================
Responsibilities:
  1. Configure MLflow autologging for LangChain.
  2. Manage versioned prompt templates via the MLflow Prompt Registry.
  3. Execute RAGAS-based automated evaluation via mlflow.genai.evaluate.
  4. Provide request-scoped callbacks to prevent cross-tenant trace bleed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlflow
import mlflow.langchain
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.config import Settings, get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MLflow Bootstrap
# ---------------------------------------------------------------------------


def configure_mlflow(settings: Settings | None = None) -> None:
    """
    Set up the MLflow tracking URI and experiment, then enable
    LangChain autologging. Must be called once at application startup.
    """
    cfg = settings or get_settings()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    mlflow.langchain.autolog(
        log_models=False,          # avoid heavy model artefact logging in dev
        log_input_examples=True,
        log_model_signatures=True,
        silent=False,
    )
    logger.info(
        "mlflow_configured",
        tracking_uri=cfg.mlflow_tracking_uri,
        experiment=cfg.mlflow_experiment_name,
    )


# ---------------------------------------------------------------------------
# Prompt Registry
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are DocuQuery, an expert assistant that answers questions exclusively
based on the provided context. Follow these rules strictly:

1. Answer ONLY from the information present in the <context> block.
2. If the context does not contain sufficient information, say
   "I cannot find enough information in the provided documents to answer
    this question" — do NOT fabricate or infer beyond the context.
3. Cite the source document name at the end of each factual claim using
   the format [Source: {source}].
4. Structure complex answers using markdown bullet points or numbered lists.
5. Be concise and precise. Avoid padding or filler phrases.

<context>
{context}
</context>
"""

_PROMPT_NAME = "docuquery-system-prompt"


def register_prompt(version_alias: str = "production") -> str:
    """
    Register the system prompt in the MLflow Prompt Registry.
    Returns the URI of the registered prompt.
    """
    client = mlflow.MlflowClient()
    try:
        result = client.create_prompt(
            name=_PROMPT_NAME,
            template=_SYSTEM_PROMPT_TEMPLATE,
            tags={"domain": "rag", "version_alias": version_alias},
        )
        logger.info("prompt_registered", name=_PROMPT_NAME, version=result.version)
        return f"prompts:/{_PROMPT_NAME}/{result.version}"
    except Exception as exc:
        logger.warning("prompt_register_failed", error=str(exc))
        return _PROMPT_NAME


def load_prompt(version: str | None = None) -> str:
    """
    Load a prompt template from the MLflow Prompt Registry.
    Falls back to the hardcoded template if MLflow is unavailable.
    """
    try:
        name = f"{_PROMPT_NAME}/{version}" if version else _PROMPT_NAME
        prompt = mlflow.genai.load_prompt(name)
        return prompt.template
    except Exception as exc:
        logger.warning("prompt_load_failed", error=str(exc), fallback="hardcoded")
        return _SYSTEM_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Request-Scoped Telemetry Callback
# ---------------------------------------------------------------------------


@dataclass
class RequestTelemetry:
    """Accumulates per-request metrics without global state pollution."""

    request_id: str
    start_time: float = field(default_factory=time.monotonic)
    first_token_time: float | None = None
    total_tokens: int = 0
    llm_calls: int = 0
    retrieval_docs: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def time_to_first_token_ms(self) -> float | None:
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.start_time) * 1000

    @property
    def total_latency_ms(self) -> float:
        return (time.monotonic() - self.start_time) * 1000


class RequestScopedCallback(BaseCallbackHandler):
    """
    LangChain callback that is instantiated per-request (never globally).
    Captures TTFT, token counts, and error traces in complete isolation
    from concurrent requests.
    """

    def __init__(self, telemetry: RequestTelemetry) -> None:
        super().__init__()
        self._tel = telemetry
        self._first_token_recorded = False

    # ── LLM Events ────────────────────────────────────────────────────────────

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if not self._first_token_recorded and token:
            self._tel.first_token_time = time.monotonic()
            self._first_token_recorded = True
        self._tel.total_tokens += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._tel.llm_calls += 1
        # Extract token usage if present in llm_output
        usage = (response.llm_output or {}).get("token_usage", {})
        total = usage.get("total_tokens", 0)
        if total:
            self._tel.total_tokens = total

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._tel.errors.append(f"LLM Error: {error!s}")
        logger.error("llm_error", request_id=self._tel.request_id, error=str(error))

    # ── Retriever Events ──────────────────────────────────────────────────────

    def on_retriever_end(self, documents: list, **kwargs: Any) -> None:
        self._tel.retrieval_docs = len(documents)

    def on_retriever_error(self, error: BaseException, **kwargs: Any) -> None:
        self._tel.errors.append(f"Retriever Error: {error!s}")


# ---------------------------------------------------------------------------
# RAGAS Evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalSample:
    """One evaluation example for RAGAS scoring."""

    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str | None = None  # ground-truth answer (optional)


def run_ragas_evaluation(
    samples: list[EvalSample],
    experiment_name: str | None = None,
    run_name: str = "ragas-eval",
) -> dict[str, float]:
    """
    Execute RAGAS evaluation using mlflow.genai.evaluate.

    Metrics computed:
      - Faithfulness            (hallucination rate)
      - Context Precision @K   (signal-to-noise in retrieval ranking)
      - Answer Relevancy        (query coverage)
      - Context Recall          (retrieval completeness — requires reference)

    Results are logged to the active MLflow experiment.
    Returns a dict of {metric_name: score}.
    """
    import pandas as pd
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    logger.info("ragas_eval_starting", num_samples=len(samples))

    records = [
        {
            "user_input": s.user_input,
            "response": s.response,
            "retrieved_contexts": s.retrieved_contexts,
            "reference": s.reference or "",
        }
        for s in samples
    ]
    df = pd.DataFrame(records)

    metrics = [faithfulness, context_precision, answer_relevancy]
    if any(s.reference for s in samples):
        metrics.append(context_recall)

    from ragas import EvaluationDataset

    eval_dataset = EvaluationDataset.from_pandas(df)

    settings = get_settings()
    with mlflow.start_run(run_name=run_name, experiment_id=_get_experiment_id(settings)):
        result = evaluate(dataset=eval_dataset, metrics=metrics)
        scores = result.to_pandas().mean(numeric_only=True).to_dict()

        for metric_name, score in scores.items():
            mlflow.log_metric(metric_name, float(score))

        mlflow.log_param("num_samples", len(samples))
        mlflow.log_param("chunk_strategy", settings.chunk_strategy.value)
        mlflow.log_param("reranker", settings.reranker_backend.value)

        logger.info("ragas_eval_complete", scores=scores)
        return scores


def _get_experiment_id(settings: Settings) -> str:
    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    if experiment is None:
        return mlflow.create_experiment(settings.mlflow_experiment_name)
    return experiment.experiment_id
