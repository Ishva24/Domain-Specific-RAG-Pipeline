import pytest
from langchain_core.documents import Document
from app.vector_stores.qdrant_store import QdrantVectorStore
from app.vector_stores.chroma_store import ChromaVectorStore


def test_qdrant_vector_store_in_memory():
    store = QdrantVectorStore(collection_name="test_collection", vector_size=3)
    docs = [
        Document(page_content="Document 1 about RAG", metadata={"topic": "rag"}),
        Document(page_content="Document 2 about Agents", metadata={"topic": "agents"}),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    count = store.upsert(docs, embeddings)
    assert count == 2
    assert store.count() == 2

    # Query matching doc 1
    results = store.query([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0][0].page_content == "Document 1 about RAG"
    assert results[0][1] > 0.99


def test_chroma_vector_store_in_memory():
    store = ChromaVectorStore(collection_name="test_chroma", persist_directory=None)
    docs = [
        Document(page_content="Document A about Late Chunking", metadata={"category": "chunking"}),
        Document(page_content="Document B about Reranking", metadata={"category": "reranking"}),
    ]
    embeddings = [
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0],
    ]

    upserted = store.upsert(docs, embeddings)
    assert upserted == 2
    assert store.count() == 2

    results = store.query([0.0, 0.0, 1.0], top_k=1)
    assert len(results) == 1
    assert results[0][0].page_content == "Document B about Reranking"
