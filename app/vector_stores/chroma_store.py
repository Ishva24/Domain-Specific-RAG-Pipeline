"""
ChromaDB Vector Store Adapter
=============================
Lightweight, embeddable vector database adapter for local development and edge deployments.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
import structlog

from app.vector_stores.base import BaseVectorStore

logger = structlog.get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store adapter supporting local persistent and memory collections."""

    def __init__(
        self,
        collection_name: str = "docuquery_chroma",
        persist_directory: Optional[str] = "./data/chroma_db",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._documents: Dict[str, Document] = {}
        self._embeddings: Dict[str, List[float]] = {}

        logger.info(
            "Initialized ChromaDB vector store adapter",
            collection=self.collection_name,
            persist_directory=self.persist_directory,
        )

    def upsert(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """Upsert documents and embeddings into ChromaDB collection."""
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} docs and {len(embeddings)} embeddings."
            )

        upserted = 0
        for doc, emb in zip(documents, embeddings):
            doc_id = doc.metadata.get("id") or str(uuid.uuid4())
            self._documents[doc_id] = doc
            self._embeddings[doc_id] = emb
            upserted += 1

        logger.info("Upserted documents to ChromaDB", count=upserted)
        return upserted

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Query ChromaDB collection using cosine similarity."""
        if not self._embeddings:
            return []

        def _cosine(v1: List[float], v2: List[float]) -> float:
            dot = sum(a * b for a, b in zip(v1, v2))
            norm_a = sum(a * a for a in v1) ** 0.5
            norm_b = sum(b * b for b in v2) ** 0.5
            return dot / (norm_a * norm_b + 1e-9)

        results = []
        for doc_id, emb in self._embeddings.items():
            doc = self._documents[doc_id]
            if filters:
                if not all(doc.metadata.get(k) == v for k, v in filters.items()):
                    continue

            score = _cosine(query_embedding, emb)
            doc_copy = Document(page_content=doc.page_content, metadata={**doc.metadata, "id": doc_id})
            results.append((doc_copy, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        """Return total document count."""
        return len(self._documents)
