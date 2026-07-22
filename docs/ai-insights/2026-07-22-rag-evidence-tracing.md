# AI Insight - RAG Evidence Tracing: Debug the Answer Back to the Chunk

Date: 2026-07-22

## Core Idea

A RAG trace should make a bad answer diagnosable, not merely observable. A single request ID and a final latency number do not reveal whether a failure came from parsing, retrieval, reranking, prompt construction, model generation, or citation rendering.

For DocuQuery, make the retrieval path an explicit trace: connect the user request to the collection version, candidate chunks, ranking decisions, selected evidence, generated answer, and visible citations. The goal is to answer a practical engineering question quickly: **which decision caused this response?**

## A Useful Trace Shape

Create one root span for the query and child spans for meaningful stages:

1. `rag.query` - records a privacy-safe query fingerprint, tenant or collection ID, request mode, and trace ID.
2. `rag.retrieve` - records the retriever version, index or corpus version, top-k, filters, candidate count, and latency.
3. `rag.rerank` - records the reranker version, input and output counts, score distribution summary, and latency.
4. `rag.context` - records the chosen chunk IDs, source revisions, token budget, truncation count, and citation IDs. Do not put full document text in standard logs.
5. `gen_ai.chat` - records model and provider, request/response token counts, finish reason, latency, and error status.
6. `rag.answer` - records the answer status, citation count, unsupported-claim check result, user-feedback signal, and a link to the traceable evidence IDs.

Use the same `trace_id` across these spans, and carry only stable identifiers between services. A chunk ID should resolve to the immutable source version that was actually retrieved, not the latest version of a file with the same name.

## Evidence Record to Persist

Store a compact per-request record alongside operational telemetry. It is the bridge between online debugging and offline evaluation.

```json
{
  "trace_id": "tr_01J...",
  "corpus_version": "policies-2026-07-22",
  "retriever": {"name": "hybrid", "version": "v3", "top_k": 12},
  "selected_evidence": [
    {"chunk_id": "leave-policy#p12:c3", "rank": 1, "score": 0.84, "source_sha": "..."}
  ],
  "generation": {"model": "gpt-4.1-mini", "prompt_tokens": 814, "completion_tokens": 186},
  "answer": {"citation_ids": ["leave-policy#p12:c3"], "grounding_status": "supported"}
}
```

The record deliberately separates a retrieval score from a grounding judgment. A high similarity score is evidence of retrieval confidence, not proof that the generated claim is supported.

## Guardrails for Useful, Safe Telemetry

- Treat query text and retrieved passages as sensitive by default. Prefer fingerprints, IDs, counts, and score summaries in normal logs; use tightly controlled, short-retention debug capture only when justified.
- Version everything that changes the answer: ingestion pipeline, embedding model, chunking strategy, index, retriever, reranker, prompt template, and generator model.
- Sample successful high-volume requests, but retain every error, timeout, empty retrieval, citation mismatch, and low-grounding outcome. Sampling away failures erases the cases that need diagnosis.
- Emit a typed failure reason such as `no_candidates`, `filtered_all`, `context_over_budget`, `generation_timeout`, or `unsupported_claim`. These states make dashboards and regression tests actionable.
- Link feedback to the trace and evidence IDs. A thumbs-down without the retrieved context cannot tell you whether to fix retrieval, prompting, or the source material.

## Portfolio-Ready Debugging Flow

When an evaluator flags an incorrect answer, inspect the trace in this order:

1. Confirm the corpus and ingestion version; stale or incorrectly parsed source material is a data issue.
2. Inspect candidates and filters; a relevant chunk missing here is a retrieval issue.
3. Compare reranked candidates with selected context; an omitted strong candidate is ranking or budget pressure.
4. Compare the answer's citations with the selected chunks; a mismatch is a generation or citation-rendering issue.
5. Feed the trace back into the evaluation dataset with the failure label and expected evidence. This turns a production incident into a reproducible regression case.

This design complements the repository's quality gates and evaluation-drift notes: quality metrics show *that* a change regressed, while evidence tracing reveals *where* it regressed.

## References

- [OpenTelemetry GenAI semantic conventions registry](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/) - standardized GenAI attributes and the current path to related conventions.
- [OpenTelemetry trace semantic conventions](https://opentelemetry.io/docs/specs/semconv/general/trace/) - trace structure, span relationships, and error recording guidance.
- [RAGAS documentation](https://docs.ragas.io/) - evaluation concepts for measuring RAG quality beyond latency and availability.
