import pytest
from unittest.mock import MagicMock
from app.query_transform import (
    HyDEGenerator,
    MultiQueryDecomposer,
    QueryTransformer,
    QueryTransformStrategy,
)


def test_hyde_generator_with_mock_llm():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "This is a detailed technical hypothetical answer about RAG pipelines."
    mock_llm.invoke.return_value = mock_response

    generator = HyDEGenerator(llm_client=mock_llm)
    result = generator.generate("How does hybrid search work?")

    assert result == "This is a detailed technical hypothetical answer about RAG pipelines."
    mock_llm.invoke.assert_called_once()


def test_hyde_generator_fallback_without_llm():
    generator = HyDEGenerator(llm_client=None)
    result = generator.generate("What is late chunking?")
    assert result == "What is late chunking?"


def test_multi_query_decomposer():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        "What are the benefits of late chunking?\n"
        "How to implement late chunking in PyTorch?\n"
        "Late chunking vs sentence splitting trade-offs"
    )
    mock_llm.invoke.return_value = mock_response

    decomposer = MultiQueryDecomposer(llm_client=mock_llm, num_queries=3)
    results = decomposer.decompose("Explain late chunking.")

    assert len(results) >= 3
    assert "Explain late chunking." in results
    assert "What are the benefits of late chunking?" in results


def test_query_transformer_strategy_routing():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Hypothetical passage for test."
    mock_llm.invoke.return_value = mock_response

    transformer_none = QueryTransformer(strategy=QueryTransformStrategy.NONE)
    assert transformer_none.transform("Test query") == ["Test query"]

    transformer_hyde = QueryTransformer(strategy=QueryTransformStrategy.HYDE, llm_client=mock_llm)
    res = transformer_hyde.transform("Test query")
    assert len(res) == 2
    assert res[0] == "Test query"
    assert res[1] == "Hypothetical passage for test."
