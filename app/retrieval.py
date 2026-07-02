"""
Phase 2 — Hybrid Retrieval Engine
===================================
Architecture:
  Query → [BM25 sparse search (k=20)] ─┐
                                         ├─► RRF Fusion ─► Cross-Encoder Re-rank ─► Top-K Docs
  Query → [FAISS dense search (k=20)] ──┘

Components:
  - HybridRetriever : orchestrates BM25 + FAISS with Reciprocal Rank Fusion
  - RerankerFactory : selects FlashRank / BGE / Cohere re-ranker
  - RRFFusion       : pure RRF implementation (no external dependencies)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document

from app.config import RerankerBackend, Settings, get_settings
from app.ingestion import BM25Index, FAISSVectorStore, JinaEmbedder
from app.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


class RRFFusion:
    """
    Reciprocal Rank Fusion for combining multiple ranked lists.

    Score = Σ  weight_i / (k + rank_i)
    where k=60 (standard constant reducing the impact of high ranks).
    """

    K_CONSTANT = 60

    @staticmethod
    def fuse(
        ranked_lists: list[list[Document]],
        weights: list[float] | None = None,
    ) -> list[Document]:
        """
        Merge ranked lists into a single deduplicated list ordered by RRF score.

        Args:
            ranked_lists: Each inner list is a ranked list of Documents.
            weights: Per-list weights (must sum to 1.0). Defaults to uniform.
        """
        n = len(ranked_lists)
        if weights is None:
            weights = [1.0 / n] * n
        assert len(weights) == n, "weights length must match ranked_lists length"

        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for ranked, weight in zip(ranked_lists, weights, strict=False):
            for rank, doc in enumerate(ranked, start=1):
                key = doc.page_content[:200]  # deduplication key
                rrf_score = weight / (RRFFusion.K_CONSTANT + rank)
                scores[key] = scores.get(key, 0.0) + rrf_score
                doc_map[key] = doc

        # Sort by descending RRF score
        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys]


# ---------------------------------------------------------------------------
# Re-rankers
# ---------------------------------------------------------------------------


class BaseReranker(ABC):
    """Abstract re-ranker interface."""

    @abstractmethod
    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Return top_k documents sorted by relevance to the query."""


class FlashRankReranker(BaseReranker):
    """
    Ultra-low latency CPU re-ranker using quantized cross-encoder models.
    Sub-20ms on commodity hardware.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2") -> None:
        from flashrank import Ranker

        logger.info("loading_flashrank_model", model=model_name)
        self._ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        from flashrank import RerankRequest

        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(documents)]
        request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(request)
        top = results[:top_k]
        return [documents[r["id"]] for r in top]


class BGEReranker(BaseReranker):
    """
    Self-hosted BGE cross-encoder (GPU recommended, CPU viable for small batches).
    High quality, multilingual support.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("loading_bge_reranker", model=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        import torch

        pairs = [[query, doc.page_content] for doc in documents]
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits.squeeze(-1)
            scores = torch.sigmoid(logits).numpy().tolist()

        ranked = sorted(zip(documents, scores, strict=False), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


class CohereReranker(BaseReranker):
    """
    Cloud SaaS re-ranker via Cohere API.
    Best quality, adds 100–400ms network latency.
    """

    def __init__(self, api_key: str, model: str = "rerank-english-v3.0") -> None:
        import cohere

        self._client = cohere.Client(api_key=api_key)
        self._model = model

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        texts = [doc.page_content for doc in documents]
        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=texts,
            top_n=top_k,
        )
        return [documents[r.index] for r in response.results]


class RerankerFactory:
    """Constructs the correct re-ranker based on settings."""

    @staticmethod
    def build(settings: Settings) -> BaseReranker:
        backend = settings.reranker_backend
        if backend == RerankerBackend.FLASHRANK:
            return FlashRankReranker()
        elif backend == RerankerBackend.BGE:
            return BGEReranker()
        elif backend == RerankerBackend.COHERE:
            if not settings.cohere_api_key:
                raise ValueError("COHERE_API_KEY is required for Cohere re-ranker")
            return CohereReranker(api_key=settings.cohere_api_key)
        else:
            raise ValueError(f"Unknown reranker backend: {backend}")


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    Two-stage retrieval pipeline:

    Stage 1 (High Recall):
      - BM25 sparse search → top initial_k/2 candidates
      - FAISS dense search → top initial_k/2 candidates
      - RRF fusion with configurable BM25/dense weights

    Stage 2 (High Precision):
      - Cross-encoder re-ranking → top final_k documents
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        faiss_store: FAISSVectorStore,
        embedder: JinaEmbedder,
        reranker: BaseReranker,
        settings: Settings | None = None,
    ) -> None:
        self._bm25 = bm25_index
        self._faiss = faiss_store
        self._embedder = embedder
        self._reranker = reranker
        self._settings = settings or get_settings()

    def retrieve(self, query: str) -> list[Document]:
        """
        Execute hybrid retrieval and return the top final_k documents.

        Returns documents sorted by cross-encoder relevance score.
        """
        cfg = self._settings
        k_per_branch = cfg.initial_retrieval_k // 2

        # ── Stage 1a: BM25 Sparse ────────────────────────────────────────────
        bm25_results = self._bm25.search(query, k=k_per_branch)
        logger.debug("bm25_retrieved", count=len(bm25_results))

        # ── Stage 1b: FAISS Dense ────────────────────────────────────────────
        query_vec = self._embedder.embed_query(query)
        faiss_results_raw = self._faiss.search(query_vec, k=k_per_branch)
        faiss_results = [doc for doc, _ in faiss_results_raw]
        logger.debug("faiss_retrieved", count=len(faiss_results))

        # ── Stage 1c: RRF Fusion ─────────────────────────────────────────────
        fused = RRFFusion.fuse(
            ranked_lists=[bm25_results, faiss_results],
            weights=[cfg.bm25_weight, cfg.dense_weight],
        )
        logger.info(
            "rrf_fusion_done",
            fused_count=len(fused),
            bm25_weight=cfg.bm25_weight,
            dense_weight=cfg.dense_weight,
        )

        # ── Stage 2: Cross-Encoder Re-ranking ────────────────────────────────
        reranked = self._reranker.rerank(query, fused, top_k=cfg.final_rerank_k)
        logger.info("reranking_done", final_count=len(reranked))

        return reranked

    def retrieve_with_metadata(self, query: str) -> dict[str, Any]:
        """
        Same as retrieve() but also returns diagnostic metadata for logging.
        """
        cfg = self._settings
        k_per_branch = cfg.initial_retrieval_k // 2

        bm25_results = self._bm25.search(query, k=k_per_branch)
        query_vec = self._embedder.embed_query(query)
        faiss_results_raw = self._faiss.search(query_vec, k=k_per_branch)
        faiss_results = [doc for doc, _ in faiss_results_raw]
        faiss_scores = [score for _, score in faiss_results_raw]

        fused = RRFFusion.fuse(
            ranked_lists=[bm25_results, faiss_results],
            weights=[cfg.bm25_weight, cfg.dense_weight],
        )
        reranked = self._reranker.rerank(query, fused, top_k=cfg.final_rerank_k)

        return {
            "documents": reranked,
            "bm25_count": len(bm25_results),
            "faiss_count": len(faiss_results),
            "fused_count": len(fused),
            "final_count": len(reranked),
            "top_faiss_scores": faiss_scores[:5],
        }


# ---------------------------------------------------------------------------
# Retriever Factory (dependency injection helper)
# ---------------------------------------------------------------------------


class RetrieverFactory:
    """
    Builds and caches a fully configured HybridRetriever.
    Designed for use as a FastAPI dependency.
    """

    _instance: HybridRetriever | None = None

    @classmethod
    def get(cls, settings: Settings | None = None) -> HybridRetriever:
        if cls._instance is None:
            cfg = settings or get_settings()
            from pathlib import Path

            faiss_path = Path(cfg.faiss_index_path)
            bm25_path = faiss_path / "bm25_index.pkl"

            faiss_store = FAISSVectorStore.load(faiss_path)
            bm25_index = BM25Index.load(bm25_path)
            embedder = JinaEmbedder(
                model_name=cfg.embedding_model,
                max_seq_len=cfg.embedding_max_seq_len,
            )
            reranker = RerankerFactory.build(cfg)

            cls._instance = HybridRetriever(
                bm25_index=bm25_index,
                faiss_store=faiss_store,
                embedder=embedder,
                reranker=reranker,
                settings=cfg,
            )
            logger.info("hybrid_retriever_initialized", backend=cfg.vector_store_backend.value)

        return cls._instance
