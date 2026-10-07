"""
Unit and integration tests for the unified IntegratedRAGPipeline.
"""

from unittest.mock import MagicMock
import pytest
from app.pipeline import IntegratedRAGPipeline, PipelineConfig


class MockDocument:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


def test_pipeline_execution_flow():
    config = PipelineConfig(
        use_query_transform=False,
        use_guardrails=True,
        retrieval_k=2,
    )
    pipeline = IntegratedRAGPipeline(config=config)

    # Mock retriever
    mock_docs = [
        MockDocument("FastAPI callbacks should be registered request-scoped.", {"source": "doc1.txt"}),
        MockDocument("Global callbacks cause cross-tenant trace bleed.", {"source": "doc2.txt"}),
    ]
    retriever_fn = MagicMock(return_value=mock_docs)

    # Mock LLM
    llm_fn = MagicMock(return_value="Global callbacks cause cross-tenant trace bleed. Always use request-scoped callbacks.")

    result = pipeline.execute(
        query="Why avoid global callbacks?",
        retriever_fn=retriever_fn,
        llm_fn=llm_fn,
    )

    assert result.query == "Why avoid global callbacks?"
    assert len(result.retrieved_documents) == 2
    assert "cross-tenant trace bleed" in result.answer
    assert result.guardrail_report is not None
    assert result.guardrail_report["faithfulness_score"] > 0.0
    assert result.execution_time_ms > 0.0


def test_pipeline_query_deduplication():
    config = PipelineConfig(
        use_query_transform=True,
        transform_strategy="none",
        retrieval_k=5,
    )
    pipeline = IntegratedRAGPipeline(config=config)

    # Retriever returning duplicate docs across calls
    retriever_fn = MagicMock(return_value=[
        MockDocument("Identical content A", {"source": "doc1.txt"}),
        MockDocument("Identical content A", {"source": "doc1.txt"}),
    ])
    llm_fn = MagicMock(return_value="Answer.")

    result = pipeline.execute("test", retriever_fn, llm_fn)
    assert len(result.retrieved_documents) == 1
