# AI Insight - Trace-to-Eval Loop for RAG

Date: 2026-07-21

## Core Idea

A production RAG trace should not end as an incident record. It should be promotable into evaluation data. When a response is slow, poorly grounded, missing a citation, or retrieves the wrong chunk, the trace already contains most of the context needed to create a regression case. The missing step is a disciplined promotion path from trace to labeled example.

For DocuQuery, this means using request traces to connect user questions, retrieval choices, reranker decisions, selected evidence, and final answers to a reusable evaluation slice. The goal is not to archive every trace forever. The goal is to turn the most instructive traces into a compact, versioned quality dataset.

## What to Preserve from the Trace

To make a trace reusable for evaluation, keep the fields that explain both retrieval and generation behavior:

- request text or a privacy-safe surrogate
- corpus or index version
- chunking strategy and retriever configuration
- top-k candidate chunk identifiers
- reranked evidence identifiers
- final answer and citations
- latency, status, and failure type
- human feedback or reviewer label when available

This is enough to reproduce most failure classes without storing the entire source corpus inside the evaluation record.

## Promotion Flow

1. Capture a trace for every request and keep retrieval plus generation spans linked under one trace identifier.
2. Search for traces that matter: explicit user thumbs-down, unsupported claims, slow responses, empty retrieval, or low-quality citations.
3. Redact or minimize sensitive payloads before reuse.
4. Add evaluator labels such as `expected_answer`, `expected_evidence_ids`, or `failure_reason`.
5. Store the promoted record in a versioned evaluation dataset and rerun it after retrieval, prompt, or model changes.

This turns production traffic into a steady source of realistic evaluation coverage instead of relying only on hand-authored benchmark questions.

## Sampling Rules That Actually Help

- Promote failures more aggressively than successes.
- Keep hard edge cases even when they are rare.
- Preserve the retrieval metadata that explains why the answer failed.
- Group traces by corpus version so evaluation results remain interpretable after reindexing.
- Retire or relabel stale cases when the source corpus materially changes.

## Minimal Promoted Record

```json
{
  "trace_id": "tr_01J...",
  "query": "What retention period applies to audit logs?",
  "corpus_version": "kb_2026_07_21",
  "retrieval": {
    "chunk_strategy": "late_chunking",
    "candidate_ids": ["policy.pdf#p12", "policy.pdf#p18"],
    "selected_evidence_ids": ["policy.pdf#p18"]
  },
  "answer": {
    "text": "Audit logs must be retained for 90 days.",
    "citations": ["policy.pdf#p18"]
  },
  "label": {
    "failure_reason": "unsupported_claim",
    "expected_evidence_ids": ["policy.pdf#p12"]
  }
}
```

The important distinction is that the trace is operational data, while the promoted record is a deliberate test case with labels attached.

## Why This Matters

RAG quality work gets faster when debugging and evaluation stop being separate activities. Traces explain what happened. Evaluations tell you whether the system is improving. A trace-to-eval loop connects the two, which is exactly what a production portfolio project should demonstrate.

## References

- [MLflow Trace Concepts](https://mlflow.org/docs/latest/genai/concepts/trace/) - traces as structured execution records with metadata and spans.
- [MLflow Search Traces](https://mlflow.org/docs/latest/genai/tracing/search-traces/) - filtering traces by status, latency, metadata, span type, and assessments.
- [Ragas Testset Generation](https://docs.ragas.io/en/stable/concepts/test_data_generation/) - maintaining varied, continually updated evaluation datasets for AI applications.
