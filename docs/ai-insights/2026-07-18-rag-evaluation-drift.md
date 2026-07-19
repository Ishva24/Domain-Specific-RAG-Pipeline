# AI Insight - RAG Evaluation Drift

Date: 2026-07-18

## Core Idea

RAG systems can drift even when the application code does not change. The document set evolves, embeddings are regenerated, chunking parameters shift, user questions become more specific, and prompt templates accumulate small edits. Each change can alter retrieval quality before anyone notices answer quality dropping.

Evaluation drift is the gap between what the RAG system was validated to do and what it is currently doing in production-like conditions. For DocuQuery, this is especially important because the system combines late chunking, hybrid BM25 plus dense retrieval, reranking, and RAGAS-style answer evaluation. A small change in one layer can create downstream answer changes that look like LLM behavior but actually started in retrieval.

## Drift Signals to Track

1. Retrieval recall drift
   - Golden questions should continue retrieving the expected source chunks.
   - A drop often means documents, chunks, embeddings, or metadata changed.

2. Context precision drift
   - Top-k results should stay focused on useful evidence.
   - More irrelevant chunks means the prompt gets noisier even if recall remains high.

3. Rank drift
   - Expected evidence should not slide far down the ranking without a reason.
   - This is a strong signal that embedding model, reranker, or fusion weights changed behavior.

4. Citation drift
   - Answers should cite stable and relevant sources.
   - Missing or weaker citations make the system harder to trust.

5. Answer faithfulness drift
   - Generated answers should remain grounded in the selected context.
   - Faithfulness drift can be caused by prompt edits, model changes, or weak retrieved context.

## Practical Evaluation Loop

A production RAG project should keep a small golden dataset and rerun it regularly. The loop can be simple:

- run retrieval-only checks first
- run answer-level checks second
- compare scores against the previous baseline
- flag regressions before changing prompts or model settings
- record the dataset version, embedding model, chunking strategy, and retrieval configuration

This makes RAG evaluation more like CI: not a one-time demo score, but a living quality contract.

## Suggested Implementation Slice

Add an evaluation report that stores each run as JSON with:

- `run_date`
- `dataset_version`
- `chunk_strategy`
- `embedding_model`
- `retrieval_top_k`
- `recall_at_k`
- `precision_at_k`
- `mean_reciprocal_rank`
- `faithfulness_score`
- `answer_relevancy_score`

A future dashboard can plot these metrics over time and show when retrieval or generation quality starts drifting.

## Why This Matters

RAG quality is not static. A system that measured well last week can weaken after new documents, updated embeddings, or prompt changes. Tracking drift makes the pipeline more reliable and gives engineers a clear place to debug before blaming the LLM.
