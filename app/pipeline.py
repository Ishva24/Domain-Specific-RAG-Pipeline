"""
Unified End-to-End Pipeline for Domain-Specific RAG.

Integrates:
1. Query Transformation (HyDE & Multi-Query decomposition)
2. Hardware Acceleration & Device Management (CUDA, MPS, CPU)
3. Hybrid Retrieval & Reranking
4. Real-time Grounding & Hallucination Guardrails
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rag_pipeline")


@dataclass
class PipelineConfig:
    use_query_transform: bool = True
    transform_strategy: str = "all"  # "none", "hyde", "multi_query", "all"
    use_guardrails: bool = True
    min_faithfulness_score: float = 0.70
    retrieval_k: int = 5
    device: Optional[str] = None


@dataclass
class PipelineOutput:
    query: str
    transformed_queries: List[str]
    retrieved_documents: List[Dict[str, Any]]
    answer: str
    guardrail_report: Optional[Dict[str, Any]] = None
    device_info: Optional[Dict[str, Any]] = None
    execution_time_ms: float = 0.0


class IntegratedRAGPipeline:
    """End-to-end RAG orchestrator unifying transformation, retrieval, generation, and validation."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._init_subsystems()

    def _init_subsystems(self) -> None:
        # Lazy imports to support lightweight environments
        try:
            from app.accel import get_device_manager
            self.device_mgr = get_device_manager()
            self.device_info = self.device_mgr.get_device_info().to_dict()
        except ImportError:
            self.device_mgr = None
            self.device_info = {"device": "cpu", "precision": "fp32"}

        try:
            from app.query_transform import QueryTransformer
            self.query_transformer = QueryTransformer()
        except ImportError:
            self.query_transformer = None

        try:
            from app.guardrails import HallucinationGuardrail
            self.guardrail = HallucinationGuardrail()
        except ImportError:
            self.guardrail = None

    def transform_query(self, query: str) -> List[str]:
        """Applies HyDE and sub-query decomposition to expand the retrieval space."""
        if not self.config.use_query_transform or not self.query_transformer:
            return [query]

        try:
            expanded = self.query_transformer.transform(query, strategy=self.config.transform_strategy)
            # Ensure the original query is always retained
            if query not in expanded:
                expanded.insert(0, query)
            return expanded
        except Exception as ex:
            logger.warning(f"Query transformation failed, falling back to original query: {ex}")
            return [query]

    def validate_answer(self, answer: str, source_texts: List[str]) -> Optional[Dict[str, Any]]:
        """Evaluates answer grounding and detects hallucinations against retrieved source texts."""
        if not self.config.use_guardrails or not self.guardrail:
            return None

        try:
            report = self.guardrail.evaluate(answer, source_texts)
            return report.to_dict()
        except Exception as ex:
            logger.warning(f"Hallucination guardrail evaluation failed: {ex}")
            return None

    def execute(
        self,
        query: str,
        retriever_fn: Any,
        llm_fn: Any,
    ) -> PipelineOutput:
        """Executes the full RAG pipeline lifecycle."""
        import time
        start_ts = time.perf_counter()

        # Step 1: Query Transformation
        transformed_queries = self.transform_query(query)

        # Step 2: Multi-Query Retrieval & Deduplication
        retrieved_docs: List[Dict[str, Any]] = []
        seen_texts = set()

        for q in transformed_queries:
            docs = retriever_fn(q, k=self.config.retrieval_k)
            for doc in docs:
                content = getattr(doc, "page_content", str(doc))
                if content not in seen_texts:
                    seen_texts.add(content)
                    metadata = getattr(doc, "metadata", {})
                    retrieved_docs.append({"content": content, "metadata": metadata})

        # Cap to target K
        retrieved_docs = retrieved_docs[:self.config.retrieval_k]
        source_texts = [d["content"] for d in retrieved_docs]

        # Step 3: Synthesis via LLM
        answer = llm_fn(query=query, context="\n\n".join(source_texts))

        # Step 4: Guardrail Grounding Verification
        guardrail_report = self.validate_answer(answer, source_texts)

        elapsed_ms = (time.perf_counter() - start_ts) * 1000.0

        return PipelineOutput(
            query=query,
            transformed_queries=transformed_queries,
            retrieved_documents=retrieved_docs,
            answer=answer,
            guardrail_report=guardrail_report,
            device_info=self.device_info,
            execution_time_ms=elapsed_ms,
        )
