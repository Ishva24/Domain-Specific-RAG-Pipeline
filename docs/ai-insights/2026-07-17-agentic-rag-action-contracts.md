# AI Insight - Action Contracts for Agentic RAG

Date: 2026-07-17

## Core Idea

Agentic RAG systems should not treat retrieval as a loose helper step. When an agent can choose tools, rewrite queries, call retrievers, summarize evidence, and trigger actions, every step needs an explicit action contract: what the step is allowed to do, what evidence it must produce, and what failure state it returns.

A plain RAG pipeline usually answers one user question. An agentic RAG pipeline may decompose the question into tasks, choose different retrievers, validate sources, and decide whether it has enough evidence to respond. That extra autonomy is useful, but it also creates risk if the agent silently moves from evidence gathering into unsupported claims.

## Practical Action Contracts

1. Query planning contract
   - Input: user question and optional conversation context.
   - Output: one or more retrieval queries plus the reason each query is needed.
   - Failure state: question is ambiguous or lacks enough scope.

2. Retrieval contract
   - Input: planned query, retrieval strategy, and top-k value.
   - Output: ranked chunks with source metadata and retrieval scores.
   - Failure state: no context crosses the minimum relevance threshold.

3. Evidence selection contract
   - Input: retrieved chunks.
   - Output: selected evidence snippets with source identifiers.
   - Failure state: context is contradictory, stale, duplicated, or insufficient.

4. Answer generation contract
   - Input: selected evidence only.
   - Output: answer, citations, and unresolved assumptions.
   - Failure state: abstain or ask a clarifying question.

5. Action execution contract
   - Input: user-approved action and verified evidence.
   - Output: action result plus audit record.
   - Failure state: missing approval, unsafe tool scope, or policy denial.

## Why This Matters

Agentic RAG is strongest when it can say no. A system that can detect weak retrieval, missing evidence, or unsafe action scope is more trustworthy than one that always produces a confident answer. Action contracts make each step inspectable, testable, and easier to monitor in production.

## Suggested Implementation Slice

Add a lightweight `RetrievalDecision` or `AgentStepDecision` type to the pipeline. Each agent step should return:

- `status`: `success`, `needs_clarification`, `insufficient_evidence`, or `blocked`
- `reason`: short explanation for the decision
- `evidence`: source ids or chunk ids used
- `next_action`: the next allowed step

This creates a clean bridge between RAG evaluation, guardrails, and future MCP or agent workflows.
