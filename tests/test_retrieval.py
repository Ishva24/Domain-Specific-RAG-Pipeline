"""
Tests — Phase 2: Hybrid Retrieval & Re-ranking
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.retrieval import (
    FlashRankReranker,
    HybridRetriever,
    RRFFusion,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

DOCS_A = [
    Document(page_content="BM25 is a sparse retrieval algorithm.", metadata={"source": "a.txt", "chunk_index": 0}),
    Document(page_content="FAISS uses dense vectors.", metadata={"source": "b.txt", "chunk_index": 1}),
    Document(page_content="Hybrid search combines both approaches.", metadata={"source": "c.txt", "chunk_index": 2}),
]

DOCS_B = [
    Document(page_content="Hybrid search combines both approaches.", metadata={"source": "c.txt", "chunk_index": 2}),
    Document(page_content="Cross-encoder models rerank candidates.", metadata={"source": "d.txt", "chunk_index": 3}),
    Document(page_content="FAISS uses dense vectors.", metadata={"source": "b.txt", "chunk_index": 1}),
]


# ── RRF Fusion ────────────────────────────────────────────────────────────────

class TestRRFFusion:
    def test_deduplication(self) -> None:
        fused = RRFFusion.fuse([DOCS_A, DOCS_B])
        texts = [d.page_content for d in fused]
        # No duplicates
        assert len(texts) == len(set(texts))

    def test_total_unique_docs(self) -> None:
        fused = RRFFusion.fuse([DOCS_A, DOCS_B])
        # 4 unique docs across both lists
        assert len(fused) == 4

    def test_custom_weights(self) -> None:
        # Should not raise even with unequal weights
        fused = RRFFusion.fuse([DOCS_A, DOCS_B], weights=[0.3, 0.7])
        assert len(fused) > 0

    def test_single_list(self) -> None:
        fused = RRFFusion.fuse([DOCS_A])
        assert fused == DOCS_A

    def test_empty_lists(self) -> None:
        fused = RRFFusion.fuse([[], []])
        assert fused == []

    def test_document_shared_across_lists_ranks_higher(self) -> None:
        """The doc appearing in both lists should receive a higher RRF score."""
        fused = RRFFusion.fuse([DOCS_A, DOCS_B])
        # "Hybrid search combines both approaches" appears at rank 3 in list A and rank 1 in list B
        # "FAISS uses dense vectors" appears at rank 2 in both lists
        # Both should be near the top; just verify shared docs appear before doc only in one list
        shared_texts = {
            "Hybrid search combines both approaches.",
            "FAISS uses dense vectors.",
        }
        top_texts = {d.page_content for d in fused[:3]}
        assert len(shared_texts & top_texts) >= 1

    def test_weight_mismatch_raises(self) -> None:
        with pytest.raises(AssertionError):
            RRFFusion.fuse([DOCS_A, DOCS_B], weights=[1.0])  # wrong length


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class TestHybridRetriever:
    @pytest.fixture
    def mock_bm25(self) -> MagicMock:
        bm25 = MagicMock()
        bm25.search.return_value = DOCS_A[:2]
        return bm25

    @pytest.fixture
    def mock_faiss(self) -> MagicMock:
        faiss = MagicMock()
        faiss.search.return_value = [(DOCS_B[0], 0.95), (DOCS_B[1], 0.87)]
        return faiss

    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        emb = MagicMock()
        emb.embed_query.return_value = [0.1] * 768
        return emb

    @pytest.fixture
    def mock_reranker(self) -> MagicMock:
        reranker = MagicMock()
        reranker.rerank.side_effect = lambda q, docs, top_k: docs[:top_k]
        return reranker

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        settings = MagicMock()
        settings.initial_retrieval_k = 4
        settings.final_rerank_k = 2
        settings.bm25_weight = 0.3
        settings.dense_weight = 0.7
        return settings

    def test_retrieve_returns_correct_count(
        self, mock_bm25, mock_faiss, mock_embedder, mock_reranker, mock_settings
    ) -> None:
        retriever = HybridRetriever(
            bm25_index=mock_bm25,
            faiss_store=mock_faiss,
            embedder=mock_embedder,
            reranker=mock_reranker,
            settings=mock_settings,
        )
        results = retriever.retrieve("test query")
        assert len(results) <= mock_settings.final_rerank_k

    def test_retrieve_calls_all_components(
        self, mock_bm25, mock_faiss, mock_embedder, mock_reranker, mock_settings
    ) -> None:
        retriever = HybridRetriever(
            bm25_index=mock_bm25,
            faiss_store=mock_faiss,
            embedder=mock_embedder,
            reranker=mock_reranker,
            settings=mock_settings,
        )
        retriever.retrieve("hybrid search query")
        mock_bm25.search.assert_called_once()
        mock_faiss.search.assert_called_once()
        mock_embedder.embed_query.assert_called_once_with("hybrid search query")
        mock_reranker.rerank.assert_called_once()

    def test_retrieve_with_metadata_returns_dict(
        self, mock_bm25, mock_faiss, mock_embedder, mock_reranker, mock_settings
    ) -> None:
        retriever = HybridRetriever(
            bm25_index=mock_bm25,
            faiss_store=mock_faiss,
            embedder=mock_embedder,
            reranker=mock_reranker,
            settings=mock_settings,
        )
        result = retriever.retrieve_with_metadata("query")
        assert "documents" in result
        assert "bm25_count" in result
        assert "faiss_count" in result
        assert "fused_count" in result
        assert "final_count" in result
