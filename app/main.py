"""
Phase 3 — FastAPI Streaming Application
=========================================
Endpoints:
  POST /api/v1/query         — SSE-streaming RAG query
  POST /api/v1/ingest        — Trigger document ingestion
  GET  /api/v1/health        — Health + readiness probe
  POST /api/v1/evaluate      — Trigger RAGAS evaluation run
  GET  /api/v1/metrics       — Retrieve recent telemetry summary

Design decisions:
  - Retrieval is executed BEFORE streaming begins (segregated from generation).
  - LangChain is invoked with request-scoped RunnableConfig callbacks.
  - EventSourceResponse (sse-starlette) for robust SSE with ping support.
  - Custom TTFTMiddleware tracks Time-to-First-Token per request.
  - All Pydantic models have strict validation.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

import mlflow
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.config import Settings, get_settings
from app.logging_config import configure_logging, get_logger
from app.mlops import (
    EvalSample,
    RequestScopedCallback,
    RequestTelemetry,
    configure_mlflow,
    load_prompt,
    run_ragas_evaluation,
)
from app.retrieval import HybridRetriever, RetrieverFactory

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# In-memory telemetry ring-buffer (last 500 requests)
# ---------------------------------------------------------------------------
_telemetry_log: deque[dict[str, Any]] = deque(maxlen=500)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    cfg = get_settings()

    logger.info("application_starting", env=cfg.app_env)
    configure_mlflow(cfg)

    # Pre-warm the retriever (loads FAISS index + BM25 from disk)
    try:
        RetrieverFactory.get(cfg)
        logger.info("retriever_warmed_up")
    except FileNotFoundError:
        logger.warning("index_not_found_run_ingest_first")

    yield

    logger.info("application_shutdown")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()

    app = FastAPI(
        title="DocuQuery RAG API",
        description=(
            "Next-Generation Domain-Specific RAG pipeline — "
            "Late Chunking · Hybrid BM25+FAISS · Cross-Encoder Re-ranking · "
            "SSE Streaming · MLflow + RAGAS Evaluation"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── TTFT Middleware ────────────────────────────────────────────────────────
    app.add_middleware(TTFTMiddleware)

    return app


app = create_app()


# ---------------------------------------------------------------------------
# TTFT Middleware
# ---------------------------------------------------------------------------


class TTFTMiddleware:
    """
    Wraps StreamingResponse to intercept the first chunk and log TTFT.
    Adds X-Request-ID and X-TTFT-Ms response headers.
    """

    def __init__(self, app_: Any) -> None:
        self.app = app_

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_start = time.monotonic()
        first_chunk_sent = False
        ttft_ms: float | None = None
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        async def send_wrapper(message: Any) -> None:
            nonlocal first_chunk_sent, ttft_ms
            if message["type"] == "http.response.body":
                body = message.get("body", b"")
                if body and not first_chunk_sent:
                    ttft_ms = (time.monotonic() - request_start) * 1000
                    first_chunk_sent = True
                    logger.info("ttft_recorded", ttft_ms=round(ttft_ms, 2), request_id=request_id)
            elif message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"x-request-id"] = request_id.encode()
                message = {**message, "headers": list(headers.items())}
            await send(message)

        await self.app(scope, receive, send_wrapper)


# ---------------------------------------------------------------------------
# Pydantic Request / Response schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="User question")
    stream: bool = Field(default=True, description="Enable SSE streaming")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to return")
    prompt_version: str | None = Field(default=None, description="MLflow prompt registry version")


class SourceDocument(BaseModel):
    content: str
    source: str
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    request_id: str
    latency_ms: float


class IngestRequest(BaseModel):
    source_directory: str = Field(..., description="Absolute path to document directory")


class EvalRequest(BaseModel):
    samples: list[dict[str, Any]] = Field(
        ..., description="List of {user_input, response, retrieved_contexts, reference?}"
    )
    run_name: str = Field(default="ragas-eval")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_retriever() -> HybridRetriever:
    try:
        return RetrieverFactory.get()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retriever not available: {exc}. Run /api/v1/ingest first.",
        ) from exc


def get_llm(settings: Settings = Depends(get_settings)) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        streaming=True,
        temperature=0.1,
        max_tokens=settings.max_context_tokens,
    )


# ---------------------------------------------------------------------------
# Helper: build context string from retrieved documents
# ---------------------------------------------------------------------------


def _format_context(documents: list, max_tokens: int = 4096) -> tuple[str, list[SourceDocument]]:
    """Concatenate document content into a context block, respecting token budget."""
    parts: list[str] = []
    sources: list[SourceDocument] = []
    char_budget = max_tokens * 4  # rough char→token heuristic

    for doc in documents:
        snippet = doc.page_content.strip()
        if not snippet:
            continue
        parts.append(f"[{doc.metadata.get('source', 'unknown')}]\n{snippet}")
        sources.append(
            SourceDocument(
                content=snippet[:500],
                source=doc.metadata.get("source", "unknown"),
                chunk_index=doc.metadata.get("chunk_index", 0),
            )
        )
        char_budget -= len(snippet)
        if char_budget <= 0:
            break

    return "\n\n---\n\n".join(parts), sources


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", tags=["System"])
async def health() -> JSONResponse:
    """Liveness + readiness probe."""
    return JSONResponse(
        {
            "status": "ok",
            "version": "1.0.0",
            "retriever_ready": RetrieverFactory._instance is not None,
        }
    )


@app.post("/api/v1/ingest", tags=["Ingestion"])
async def ingest(request: IngestRequest) -> JSONResponse:
    """
    Trigger document ingestion for a given directory.
    Runs synchronously in a thread-pool to avoid blocking the event loop.
    """
    from app.ingestion import IngestionPipeline

    def _run_ingestion() -> dict:
        pipeline = IngestionPipeline()
        return pipeline.run(request.source_directory)

    try:
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(None, _run_ingestion)
        # Reset cached retriever so it reloads the new index
        RetrieverFactory._instance = None
        RetrieverFactory.get()
        return JSONResponse({"status": "success", **summary})
    except Exception as exc:
        logger.error("ingestion_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/query", tags=["RAG"])
async def query(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    llm: ChatOpenAI = Depends(get_llm),
    settings: Settings = Depends(get_settings),
) -> Any:
    """
    Main RAG endpoint.
    - Non-streaming: returns complete QueryResponse JSON.
    - Streaming:     returns Server-Sent Events (text/event-stream).

    Retrieval is always executed before streaming begins.
    """
    request_id = str(uuid.uuid4())
    telemetry = RequestTelemetry(request_id=request_id)
    callback = RequestScopedCallback(telemetry)

    # ── Phase A: Retrieval (always synchronous, before any streaming) ─────────
    t0 = time.monotonic()
    try:
        retrieved_docs = await asyncio.get_event_loop().run_in_executor(
            None, retriever.retrieve, request.query
        )
    except Exception as exc:
        logger.error("retrieval_failed", error=str(exc), request_id=request_id)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}") from exc

    retrieval_ms = (time.monotonic() - t0) * 1000
    telemetry.retrieval_docs = len(retrieved_docs)
    logger.info("retrieval_complete", ms=round(retrieval_ms, 2), docs=len(retrieved_docs))

    # ── Phase B: Prompt construction ──────────────────────────────────────────
    system_template = load_prompt(request.prompt_version)
    context_str, source_docs = _format_context(retrieved_docs, settings.max_context_tokens)
    system_content = system_template.format(context=context_str)

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=request.query),
    ]

    runnable_config = RunnableConfig(callbacks=[callback])

    # ── Phase C: Generation ───────────────────────────────────────────────────
    if not request.stream:
        # Non-streaming path
        response = await llm.ainvoke(messages, config=runnable_config)
        total_ms = (time.monotonic() - t0) * 1000
        _log_telemetry(telemetry, total_ms)

        return QueryResponse(
            answer=response.content,
            sources=source_docs,
            request_id=request_id,
            latency_ms=round(total_ms, 2),
        )

    # Streaming path — SSE
    async def event_generator() -> AsyncGenerator[dict, None]:
        full_answer = ""
        try:
            async for chunk in llm.astream(messages, config=runnable_config):
                token = chunk.content
                if token:
                    full_answer += token
                    yield {"event": "token", "data": token}
                    await asyncio.sleep(0)  # yield control to the event loop

            # Emit source metadata as terminal JSON event
            sources_payload = json.dumps(
                {
                    "sources": [s.model_dump() for s in source_docs],
                    "request_id": request_id,
                    "latency_ms": round((time.monotonic() - t0) * 1000, 2),
                }
            )
            yield {"event": "sources", "data": sources_payload}

        except asyncio.CancelledError:
            logger.warning("stream_cancelled", request_id=request_id)
        except Exception as exc:
            logger.error("stream_error", error=str(exc), request_id=request_id)
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}
        finally:
            total_ms = (time.monotonic() - t0) * 1000
            _log_telemetry(telemetry, total_ms)
            yield {"event": "done", "data": ""}

    return EventSourceResponse(
        event_generator(),
        ping=15,  # Send keep-alive pings every 15 seconds
        ping_message_factory=lambda: {"event": "ping", "data": ""},  # type: ignore[return-value]
    )


@app.post("/api/v1/evaluate", tags=["MLOps"])
async def evaluate(request: EvalRequest) -> JSONResponse:
    """Trigger a RAGAS evaluation run against the MLflow tracking server."""
    try:
        samples = [
            EvalSample(
                user_input=s["user_input"],
                response=s["response"],
                retrieved_contexts=s.get("retrieved_contexts", []),
                reference=s.get("reference"),
            )
            for s in request.samples
        ]
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None, lambda: run_ragas_evaluation(samples, run_name=request.run_name)
        )
        return JSONResponse({"status": "success", "scores": scores})
    except Exception as exc:
        logger.error("evaluation_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/metrics", tags=["System"])
async def metrics() -> JSONResponse:
    """Return aggregated telemetry from the last 500 requests."""
    if not _telemetry_log:
        return JSONResponse({"message": "No telemetry data yet."})

    ttfts = [r["ttft_ms"] for r in _telemetry_log if r.get("ttft_ms") is not None]
    latencies = [r["total_latency_ms"] for r in _telemetry_log]

    return JSONResponse(
        {
            "total_requests": len(_telemetry_log),
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "p95_latency_ms": round(_percentile(latencies, 95), 2),
            "p99_latency_ms": round(_percentile(latencies, 99), 2),
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_telemetry(telemetry: RequestTelemetry, total_ms: float) -> None:
    record = {
        "request_id": telemetry.request_id,
        "ttft_ms": telemetry.time_to_first_token_ms,
        "total_latency_ms": round(total_ms, 2),
        "total_tokens": telemetry.total_tokens,
        "retrieval_docs": telemetry.retrieval_docs,
        "errors": telemetry.errors,
    }
    _telemetry_log.append(record)
    logger.info("request_telemetry", **record)


def _percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]
