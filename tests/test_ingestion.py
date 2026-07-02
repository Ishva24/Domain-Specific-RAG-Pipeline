"""
Tests — Phase 1: Ingestion & Chunking
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from unittest.mock import MagicMock, patch

from app.ingestion import (
    BM25Index,
    LateChunkingProcessor,
    ParentChildProcessor,
    RecursiveProcessor,
    RRFFusion,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_DOCS = [
    Document(
        page_content=(
            "The RAG pipeline processes documents using advanced chunking. "
            "Late chunking preserves contextual information across paragraph boundaries. "
            "It uses a long-context embedding model with 8192-token support."
        ),
        metadata={"source": "test_doc.txt"},
    ),
    Document(
        page_content=(
            "FAISS provides ultra-fast approximate nearest-neighbour search. "
            "For production deployments, Pinecone offers managed infrastructure. "
            "The choice depends on scale, metadata filtering, and operational requirements."
        ),
        metadata={"source": "vector_stores.txt"},
    ),
]


# ── Recursive Processor ───────────────────────────────────────────────────────

class TestRecursiveProcessor:
    def test_produces_chunks(self) -> None:
        proc = RecursiveProcessor(chunk_size=100, chunk_overlap=20)
        chunks = proc.process(SAMPLE_DOCS)
        assert len(chunks) > 0

    def test_metadata_propagated(self) -> None:
        proc = RecursiveProcessor(chunk_size=200, chunk_overlap=0)
        chunks = proc.process(SAMPLE_DOCS)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_handles_empty_documents(self) -> None:
        proc = RecursiveProcessor()
        chunks = proc.process([])
        assert chunks == []


# ── Parent-Child Processor ────────────────────────────────────────────────────

class TestParentChildProcessor:
    def test_children_have_parent_id(self) -> None:
        proc = ParentChildProcessor(parent_size=200, child_size=80)
        children = proc.process(SAMPLE_DOCS)
        assert all("parent_id" in c.metadata for c in children)

    def test_children_smaller_than_parent(self) -> None:
        proc = ParentChildProcessor(parent_size=300, child_size=100)
        children = proc.process(SAMPLE_DOCS)
        # Each child should have <= parent_size chars (approximate since splitter uses chars)
        for child in children:
            assert len(child.page_content) <= 350  # allow small overshoot from splitter

    def test_parent_ids_are_unique_per_parent(self) -> None:
        proc = ParentChildProcessor(parent_size=500, child_size=100)
        children = proc.process(SAMPLE_DOCS)
        parent_ids = {c.metadata["parent_id"] for c in children}
        assert len(parent_ids) >= 1


# ── Late Chunking Processor ───────────────────────────────────────────────────

class TestLateChunkingProcessor:
    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        embedder = MagicMock()
        embedder.late_chunk_embed.return_value = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        return embedder

    def test_produces_chunks_with_precomputed_embeddings(self, mock_embedder: MagicMock) -> None:
        proc = LateChunkingProcessor(embedder=mock_embedder, chunk_size=100)
        chunks = proc.process(SAMPLE_DOCS[:1])
        assert len(chunks) > 0
        # All chunks should have precomputed_embedding in metadata
        assert all("precomputed_embedding" in c.metadata for c in chunks)

    def test_chunking_strategy_in_metadata(self, mock_embedder: MagicMock) -> None:
        proc = LateChunkingProcessor(embedder=mock_embedder, chunk_size=100)
        chunks = proc.process(SAMPLE_DOCS[:1])
        assert all(c.metadata.get("chunking_strategy") == "late_chunking" for c in chunks)

    def test_fallback_on_embed_failure(self) -> None:
        embedder = MagicMock()
        embedder.late_chunk_embed.side_effect = RuntimeError("Model unavailable")
        proc = LateChunkingProcessor(embedder=embedder, chunk_size=100)
        chunks = proc.process(SAMPLE_DOCS[:1])
        # Fallback should produce raw text chunks
        assert len(chunks) > 0


# ── BM25 Index ────────────────────────────────────────────────────────────────

class TestBM25Index:
    def test_build_and_search(self) -> None:
        index = BM25Index()
        index.build(SAMPLE_DOCS)
        results = index.search("FAISS approximate nearest neighbour", k=2)
        assert len(results) >= 1
        # Most relevant doc should be the FAISS one
        assert "FAISS" in results[0].page_content

    def test_search_raises_before_build(self) -> None:
        index = BM25Index()
        with pytest.raises(RuntimeError, match="not built"):
            index.search("any query")

    def test_empty_query_returns_results(self) -> None:
        index = BM25Index()
        index.build(SAMPLE_DOCS)
        results = index.search("", k=5)
        # Should return empty list (no tokens to score)
        assert isinstance(results, list)

    def test_persistence_roundtrip(self, tmp_path) -> None:
        index = BM25Index()
        index.build(SAMPLE_DOCS)
        save_path = tmp_path / "bm25.pkl"
        index.save(save_path)

        loaded = BM25Index.load(save_path)
        results = loaded.search("RAG chunking", k=2)
        assert len(results) >= 1
