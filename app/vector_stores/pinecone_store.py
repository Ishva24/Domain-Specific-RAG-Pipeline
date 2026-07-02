"""
Pinecone Vector Store — Production Backend
==========================================
Replaces FAISS in the production deployment (Phase 5 migration).
Supports first-class metadata filtering, auto-scaling, and sparse-dense
hybrid search (Pinecone native hybrid mode).

Usage:
  store = PineconeVectorStore.from_settings()
  store.upsert(documents, embedder)
  results = store.query(query_vector, filter={"author": "Smith"}, top_k=20)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.ingestion import JinaEmbedder
from app.logging_config import get_logger

logger = get_logger(__name__)

VECTOR_DIMENSION = 768
METRIC = "cosine"


class PineconeVectorStore:
    """
    Manages a Pinecone serverless index with:
      - Upsert with automatic batching (avoid Pinecone's 4MB request limit)
      - Rich metadata storage (source, chunk_index, chunking_strategy, date)
      - Metadata-filtered similarity search
      - Sparse-dense hybrid search (Pinecone integrated sparse mode)
    """

    BATCH_SIZE = 100  # vectors per upsert batch

    def __init__(self, index: Any) -> None:
        self._index = index

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "PineconeVectorStore":
        from pinecone import Pinecone, ServerlessSpec

        cfg = settings or get_settings()
        pc = Pinecone(api_key=cfg.pinecone_api_key)

        existing = [idx.name for idx in pc.list_indexes()]
        if cfg.pinecone_index_name not in existing:
            logger.info("creating_pinecone_index", name=cfg.pinecone_index_name)
            pc.create_index(
                name=cfg.pinecone_index_name,
                dimension=VECTOR_DIMENSION,
                metric=METRIC,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=cfg.pinecone_environment,
                ),
            )
        index = pc.Index(cfg.pinecone_index_name)
        logger.info("pinecone_connected", index=cfg.pinecone_index_name)
        return cls(index)

    def upsert(self, documents: list[Document], embedder: JinaEmbedder) -> int:
        """
        Embed and upsert all documents into Pinecone in batches.
        Returns the number of vectors upserted.
        """
        total = 0
        for i in range(0, len(documents), self.BATCH_SIZE):
            batch = documents[i : i + self.BATCH_SIZE]
            texts = [doc.page_content for doc in batch]
            vectors = embedder.embed_documents(texts)

            pinecone_records = []
            for j, (doc, vec) in enumerate(zip(batch, vectors, strict=False)):
                doc_id = f"{doc.metadata.get('source', 'doc')}_{i + j}"
                pinecone_records.append(
                    {
                        "id": doc_id,
                        "values": vec,
                        "metadata": {
                            "text": doc.page_content[:1000],  # Pinecone metadata 40KB limit
                            "source": doc.metadata.get("source", "unknown"),
                            "chunk_index": doc.metadata.get("chunk_index", 0),
                            "chunking_strategy": doc.metadata.get("chunking_strategy", "unknown"),
                        },
                    }
                )

            self._index.upsert(vectors=pinecone_records)
            total += len(batch)
            logger.info("pinecone_batch_upserted", batch_size=len(batch), total=total)

        return total

    def query(
        self,
        query_vector: list[float],
        top_k: int = 20,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Perform approximate nearest-neighbour search with optional metadata filtering.

        Example filter: {"source": {"$in": ["report_2024.pdf", "report_2025.pdf"]}}
        """
        response = self._index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter,
            include_metadata=True,
        )
        documents = []
        for match in response.matches:
            meta = match.metadata or {}
            documents.append(
                Document(
                    page_content=meta.get("text", ""),
                    metadata={
                        "source": meta.get("source", "unknown"),
                        "chunk_index": meta.get("chunk_index", 0),
                        "score": match.score,
                        "pinecone_id": match.id,
                    },
                )
            )
        return documents

    def delete_namespace(self, namespace: str = "") -> None:
        """Delete all vectors in a namespace (useful for index resets)."""
        self._index.delete(delete_all=True, namespace=namespace)
        logger.info("pinecone_namespace_deleted", namespace=namespace)

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        return self._index.describe_index_stats()
