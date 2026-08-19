import pytest
from langchain_core.documents import Document
from app.guardrails import ClaimExtractor, GroundingValidator, HallucinationGuardrail


def test_claim_extractor():
    extractor = ClaimExtractor()
    text = (
        "DocuQuery uses late chunking with Jina embeddings. "
        "It supports hybrid BM25 and dense vector search with RRF fusion. "
        "The system runs on Python 3.12."
    )
    claims = extractor.extract_claims(text)
    assert len(claims) == 3
    assert "DocuQuery uses late chunking with Jina embeddings." in claims


def test_grounding_validator_supported():
    validator = GroundingValidator(threshold=0.4)
    context_docs = [
        Document(
            page_content="Late chunking applies transformer embeddings over full documents before pooling token spans."
        )
    ]
    claim = "Late chunking applies embeddings over full documents before pooling spans."
    result = validator.verify_claim(claim, context_docs)
    assert result.is_supported is True
    assert result.confidence >= 0.4
    assert result.supporting_context_snippet is not None


def test_grounding_validator_unsupported_hallucination():
    validator = GroundingValidator(threshold=0.5)
    context_docs = [
        Document(page_content="The model classifies medical MRI scans for gliomas and meningiomas.")
    ]
    hallucinated_claim = "The system executes high frequency algorithmic stock options arbitrage."
    result = validator.verify_claim(hallucinated_claim, context_docs)
    assert result.is_supported is False
    assert result.confidence < 0.5


def test_hallucination_guardrail_report():
    guardrail = HallucinationGuardrail(threshold=0.4)
    context_docs = [
        Document(page_content="Pinecone serverless vector database indexes dense document embeddings.")
    ]
    answer = (
        "Pinecone serverless vector database indexes dense embeddings. "
        "The system was invented in 1845 on the planet Mars."
    )
    report = guardrail.evaluate(answer, context_docs)
    assert report.total_claims == 2
    assert report.supported_claims_count == 1
    assert len(report.unsupported_claims) == 1
    assert "The system was invented in 1845 on the planet Mars." in report.unsupported_claims
    assert report.faithfulness_score == 0.5
    assert report.is_grounded is False
