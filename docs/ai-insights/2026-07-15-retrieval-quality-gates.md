# AI Insight - Retrieval Quality Gates for Production RAG

Date: 2026-07-15

## Core Idea

RAG quality usually fails before generation. If retrieval brings weak, duplicated, stale, or poorly ranked context into the prompt, even a strong LLM can only produce a polished answer over shaky evidence. A production RAG system should treat retrieval as a measured contract, not an invisible preprocessing step.

For DocuQuery, the best daily improvement path is to add retrieval quality gates before expanding the generation layer. The current architecture already has the right building blocks: hybrid BM25 plus FAISS retrieval, reciprocal rank fusion, reranking, and RAGAS evaluation. The next step is to connect those pieces into a small repeatable scorecard that blocks regressions.

## Practical Quality Gates

1. Query coverage
   - Every golden query should retrieve at least one context that overlaps with the expected answer evidence.
   - Track this before reranking and after reranking to see whether the reranker helps or hurts.

2. Context precision
   - Top-k documents should be mostly useful, not just semantically adjacent.
   - A good first target is precision at 5 above 0.80 for curated evaluation samples.

3. Source diversity
   - Top-k results should avoid returning several chunks from the same narrow passage unless the query needs it.
   - This helps reduce prompt waste and improves multi-hop answers.

4. Rank stability
   - Small query rewrites should not completely reshuffle the top results.
   - This catches brittle embedding or tokenization behavior.

5. Citation readiness
   - Retrieved chunks should carry enough metadata to explain where the answer came from.
   - A chunk without source, page, section, or document identifiers is harder to trust in an API response.

## Suggested Implementation Slice

Add a lightweight retrieval evaluation command that reads `data/eval_golden_dataset.json`, runs the retriever for each query, and emits:

- recall at k
- precision at k
- average reciprocal rank
- duplicate source ratio
- missing metadata count

This can run faster than full RAGAS evaluation because it avoids LLM calls. It becomes the first CI-friendly quality gate, while RAGAS remains the deeper evaluation layer for answer-level checks.

## Why This Matters

Most RAG demos optimize for a good answer on a happy-path question. Portfolio-grade RAG work should show that the system can measure itself. A retrieval scorecard demonstrates engineering maturity because it turns "the answer looks good" into "the evidence pipeline is improving."

## Next Commit Candidate

Create `app/retrieval_eval.py` or a `docuquery retrieval-eval` CLI command that calculates these metrics from the existing golden dataset. Start with deterministic, dependency-light metrics so it can run locally and in CI without requiring API keys.
