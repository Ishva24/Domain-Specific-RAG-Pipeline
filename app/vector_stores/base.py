"""
Base Vector Store Interface
===========================
Defines the standard abstract contract for all vector store backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """Abstract base class for vector database integrations."""

    @abstractmethod
    def upsert(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """Upsert documents with their corresponding dense vector embeddings.

        Returns total count of successfully upserted documents.
        """
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Query vector store using dense vector representation.

        Returns a list of (Document, similarity_score) tuples ordered by relevance.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of vectors in the store."""
        pass
