"""
Real-Time Grounding & Hallucination Guardrail Engine
===================================================
Inspects generated LLM answers against retrieved source context passages in real-time.
Evaluates claim-level support to prevent hallucinations and compute a faithfulness score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ClaimVerificationResult:
    claim: str
    is_supported: bool
    confidence: float
    supporting_context_snippet: Optional[str] = None


@dataclass
class GuardrailReport:
    is_grounded: bool
    faithfulness_score: float
    total_claims: int
    supported_claims_count: int
    unsupported_claims: List[str]
    claims_detail: List[ClaimVerificationResult]


class ClaimExtractor:
    """Extracts atomic factual claims from generated text."""

    def extract_claims(self, text: str) -> List[str]:
        """Splits sentences and cleans markdown formatting to isolate claims."""
        clean_text = re.sub(r"\[.*?\]\(.*?\)", "", text)
        sentences = re.split(r"(?<=[.!?])\s+", clean_text.strip())
        claims = [
            s.strip()
            for s in sentences
            if len(s.strip()) > 15 and not s.strip().startswith(("#", "-", "*", ">"))
        ]
        return claims or [text.strip()]


class GroundingValidator:
    """Validates factual consistency and semantic alignment between claims and retrieved context."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def _tokenize(self, text: str) -> set[str]:
        words = re.findall(r"\b\w{3,}\b", text.lower())
        stopwords = {
            "this", "that", "with", "from", "have", "were", "which", "their",
            "about", "these", "other", "there", "would", "could", "should", "what", "where"
        }
        return {w for w in words if w not in stopwords}

    def verify_claim(
        self, claim: str, context_docs: List[Document]
    ) -> ClaimVerificationResult:
        """Verify if a single claim is grounded within the retrieved context documents."""
        claim_tokens = self._tokenize(claim)
        if not claim_tokens:
            return ClaimVerificationResult(claim=claim, is_supported=True, confidence=1.0)

        best_score = 0.0
        best_snippet = None

        for doc in context_docs:
            context_text = doc.page_content
            context_tokens = self._tokenize(context_text)

            if not context_tokens:
                continue

            overlap = len(claim_tokens.intersection(context_tokens))
            score = overlap / len(claim_tokens)

            if score > best_score:
                best_score = score
                best_snippet = context_text[:200]

        is_supported = best_score >= self.threshold

        return ClaimVerificationResult(
            claim=claim,
            is_supported=is_supported,
            confidence=round(best_score, 3),
            supporting_context_snippet=best_snippet if is_supported else None,
        )


class HallucinationGuardrail:
    """Unified guardrail to score and filter hallucinations in generated RAG responses."""

    def __init__(self, threshold: float = 0.45):
        self.extractor = ClaimExtractor()
        self.validator = GroundingValidator(threshold=threshold)

    def evaluate(
        self, answer: str, context_docs: List[Document]
    ) -> GuardrailReport:
        """Evaluate full answer against context documents and return a structured audit report."""
        if not answer.strip() or not context_docs:
            return GuardrailReport(
                is_grounded=True,
                faithfulness_score=1.0,
                total_claims=0,
                supported_claims_count=0,
                unsupported_claims=[],
                claims_detail=[],
            )

        claims = self.extractor.extract_claims(answer)
        results: List[ClaimVerificationResult] = []
        unsupported: List[str] = []

        for claim in claims:
            res = self.validator.verify_claim(claim, context_docs)
            results.append(res)
            if not res.is_supported:
                unsupported.append(claim)

        supported_count = len(claims) - len(unsupported)
        faithfulness_score = round(supported_count / len(claims), 3) if claims else 1.0
        is_grounded = faithfulness_score >= 0.7

        logger.info(
            "Guardrail verification completed",
            faithfulness_score=faithfulness_score,
            total_claims=len(claims),
            unsupported_count=len(unsupported),
            is_grounded=is_grounded,
        )

        return GuardrailReport(
            is_grounded=is_grounded,
            faithfulness_score=faithfulness_score,
            total_claims=len(claims),
            supported_claims_count=supported_count,
            unsupported_claims=unsupported,
            claims_detail=results,
        )
