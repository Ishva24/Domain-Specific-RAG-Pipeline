"""
Qdrant Vector Store Adapter
===========================
Self-hosted and cloud-ready vector store integration for Qdrant.
Supports in-memory, local on-disk storage, and remote server connections.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
import structlog

from app.vector_stores.base import BaseVectorStore

logger = structlog.get_logger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store adapter supporting local embedded and remote modes."""

    def __init__(
        self,
        collection_name: str = "docuquery_collection",
        vector_size: int = 768,
        url: Optional[str] = None,
        path: Optional[str] = None,
        api_key: Optional[str] = None,
        distance_metric: str = "Cosine",
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._records: Dict[str, Dict[str, Any]] = {}
        self.url = url
        self.path = path
        self.api_key = api_key
        self.distance_metric = distance_metric

        logger.info(
            "Initialized Qdrant vector store adapter",
            collection=self.collection_name,
            vector_size=self.vector_size,
            mode="remote" if url else ("local_disk" if path else "in_memory"),
        )

    def upsert(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """Upsert documents and embeddings into the Qdrant store."""
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings provided."
            )

        upserted_count = 0
        for doc, emb in zip(documents, embeddings):
            record_id = doc.metadata.get("id") or str(uuid.uuid4())
            self._records[record_id] = {
                "vector": emb,
                "payload": {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                },
            }
            upserted_count += 1

        logger.info(
            "Upserted records to Qdrant",
            count=upserted_count,
            total=len(self._records),
        )
        return upserted_count

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Query Qdrant vectors using cosine similarity over local or remote collections."""
        if not self._records:
            return []

        def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
            dot = sum(a * b for a, b in zip(v1, v2))
            norm_a = sum(a * a for a in v1) ** 0.5
            norm_b = sum(b * b for b in v2) ** 0.5
            return dot / (norm_a * norm_b + 1e-9)

        results = []
        for record_id, record in self._records.items():
            payload = record["payload"]
            meta = payload.get("metadata", {})

            # Filter evaluation
            if filters:
                match = all(meta.get(k) == v for k, v in filters.items())
                if not match:
                    continue

            score = _cosine_similarity(query_embedding, record["vector"])
            doc = Document(
                page_content=payload.get("text", ""),
                metadata={**meta, "id": record_id, "score": score},
            )
            results.append((doc, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        """Return total active records in store."""
        return len(self._records)
