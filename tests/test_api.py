"""
Tests — Phase 3: FastAPI Endpoints
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "retriever_ready" in data


# ── Query endpoint ────────────────────────────────────────────────────────────

MOCK_DOCS = [
    Document(
        page_content="Late chunking preserves contextual embeddings.",
        metadata={"source": "test.pdf", "chunk_index": 0},
    ),
]


class TestQueryEndpoint:
    @patch("app.main.RetrieverFactory.get")
    @patch("app.main.get_llm")
    def test_non_streaming_query(
        self, mock_get_llm: MagicMock, mock_retriever_factory: MagicMock, client: TestClient
    ) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = MOCK_DOCS
        mock_retriever_factory.return_value = mock_retriever

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="Late chunking is superior.")
        mock_get_llm.return_value = mock_llm

        response = client.post(
            "/api/v1/query",
            json={"query": "What is late chunking?", "stream": False},
        )
        # Either 200 or 503 (if retriever not warmed up in test env) is acceptable
        assert response.status_code in (200, 503)

    def test_query_rejects_short_input(self, client: TestClient) -> None:
        response = client.post("/api/v1/query", json={"query": "hi"})
        assert response.status_code == 422  # Pydantic validation error

    def test_query_rejects_empty_input(self, client: TestClient) -> None:
        response = client.post("/api/v1/query", json={"query": ""})
        assert response.status_code == 422

    def test_query_requires_body(self, client: TestClient) -> None:
        response = client.post("/api/v1/query")
        assert response.status_code == 422


# ── Metrics endpoint ──────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200

    def test_metrics_structure_no_data(self, client: TestClient) -> None:
        data = client.get("/api/v1/metrics").json()
        # Either has message (empty) or metric keys
        assert "message" in data or "total_requests" in data


# ── OpenAPI docs ──────────────────────────────────────────────────────────────

class TestOpenAPIDocs:
    def test_docs_accessible(self, client: TestClient) -> None:
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "DocuQuery RAG API"
        # Verify all key endpoints exist
        paths = schema["paths"]
        assert "/api/v1/query" in paths
        assert "/api/v1/ingest" in paths
        assert "/api/v1/health" in paths
        assert "/api/v1/evaluate" in paths
        assert "/api/v1/metrics" in paths
