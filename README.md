# DocuQuery RAG 🚀

> **Next-Generation Domain-Specific RAG Pipeline** — Production-ready, MLOps-driven architecture with Late Chunking, Hybrid BM25+FAISS Retrieval, SSE Streaming, and RAGAS Evaluation.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.16-orange.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture Overview

```
Documents
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: INGESTION PIPELINE                      │
│                                                                     │
│  DocumentLoader  ──►  Chunking Strategy  ──►  JinaEmbedder         │
│  (PDF/DOCX/TXT)        │                       (8192-token)         │
│                        ├── Late Chunking  ──────────────►  FAISS   │
│                        ├── Parent-Child                    Index   │
│                        └── Recursive                               │
│                                          ──────────────►  BM25    │
│                                                           Index   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 2: HYBRID RETRIEVAL ENGINE                   │
│                                                                     │
│  Query ──► BM25 Search (k=20)  ─────────────┐                      │
│                                              ├─► RRF Fusion         │
│  Query ──► FAISS Dense Search (k=20)  ───────┘         │           │
│                                                         ▼           │
│                                              Cross-Encoder Rerank   │
│                                              (40 → Top 5 docs)     │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                PHASE 3: FASTAPI STREAMING LAYER                     │
│                      22                                               │
│  POST /api/v1/query                                                 │
│      │                                                              │
│      ├── [Phase A] Retrieval (sync, pre-stream)                     │
│      ├── [Phase B] Prompt Construction (MLflow Prompt Registry)     │
│      └── [Phase C] LLM Streaming (SSE / EventSourceResponse)       │
│              │                                                      │
│              ├── event: token  { data: "..." }  ← per token        │
│              ├── event: sources { data: "[...]" } ← citations      │
│              └── event: done   { data: "" }                         │
│                                                                     │
│  TTFTMiddleware  →  logs Time-to-First-Token per request            │
│  RequestScopedCallback  →  isolated telemetry (no bleed)            │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                PHASE 4: MLOps & RAGAS EVALUATION                    │
│                                                                     │
│  MLflow Autologging  →  Traces per request                          │
│  Prompt Registry     →  Versioned, rollback-capable prompts         │
│  RAGAS Scorers       →  Faithfulness / Precision / Relevancy        │
│  mlflow.genai.eval   →  Logged metrics in MLflow UI                 │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│            PHASE 5: CONTAINERISATION (uv + Docker)                  │
│                                                                     │
│  Builder Stage (uv)  →  UV_COMPILE_BYTECODE + UV_LINK_MODE=copy     │
│  Runtime Stage       →  python:3.12-slim-bookworm, non-root user    │
│  Target image size   →  < 300 MB                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager (`pip install uv`)
- Docker (for containerised deployment)

### 1. Install Dependencies

```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install all dependencies
uv sync

# Activate the environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Start MLflow Tracking Server

```bash
docuquery mlflow-server --port 5000
# OR directly:
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

### 4. Ingest Your Documents

```bash
# Place your PDFs/DOCX/TXT files in a directory, then:
docuquery ingest --source ./data/documents --strategy late_chunking

# Available strategies:
#   late_chunking   — recommended for long-form, technical documents
#   parent_child    — best for multi-hop reasoning tasks
#   recursive       — fastest, good baseline
```

### 5. Start the API Server

```bash
docuquery serve --port 8000

# The API will be available at:
#   Swagger UI:  http://localhost:8000/docs
#   ReDoc:       http://localhost:8000/redoc
#   Health:      http://localhost:8000/api/v1/health
```

### 6. Build the Package

```bash
# Produce a source distribution and wheel in dist/
uv build
```

This is the fastest way to verify the project still packages cleanly before pushing documentation or code changes.

### 7. Query the RAG Pipeline

**Streaming (SSE):**
```bash
curl -N -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main chunking strategies?", "stream": true}'
```

**Non-streaming:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is late chunking?", "stream": false}'
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Liveness + readiness probe |
| `/api/v1/query` | POST | RAG query with SSE streaming |
| `/api/v1/ingest` | POST | Trigger document ingestion |
| `/api/v1/evaluate` | POST | Run RAGAS evaluation |
| `/api/v1/metrics` | GET | TTFT + latency telemetry |

---

## Running RAGAS Evaluation

```bash
# Use the provided golden dataset:
docuquery eval --samples data/eval_golden_dataset.json

# Results are logged to MLflow and printed to console:
# ┌─────────────────────┬────────┬────────┐
# │ Metric              │ Score  │ Status │
# ├─────────────────────┼────────┼────────┤
# │ faithfulness        │ 0.9312 │ ✅ PASS │
# │ context_precision   │ 0.8754 │ ✅ PASS │
# │ answer_relevancy    │ 0.8901 │ ✅ PASS │
# │ context_recall      │ 0.8200 │ ✅ PASS │
# └─────────────────────┴────────┴────────┘
```

---

## Docker Deployment

```bash
# Development (FAISS + MLflow):
docker compose up --build

# Production (adds Nginx reverse proxy):
docker compose --profile production up --build

# View logs:
docker compose logs -f api
```

---

## Testing

```bash
# Run all tests with coverage:
uv run pytest

# Run specific test module:
uv run pytest tests/test_ingestion.py -v
uv run pytest tests/test_retrieval.py -v
uv run pytest tests/test_api.py -v
```

---

## Project Structure

```
RAG/
├── app/
│   ├── __init__.py          # Package metadata
│   ├── config.py            # Pydantic-Settings (all config knobs)
│   ├── logging_config.py    # Structlog bootstrap (JSON/dev mode)
│   ├── ingestion.py         # Phase 1: Chunking + FAISS + BM25
│   ├── retrieval.py         # Phase 2: Hybrid BM25+FAISS + RRF + Reranker
│   ├── main.py              # Phase 3: FastAPI app + SSE streaming
│   ├── mlops.py             # Phase 4: MLflow + RAGAS evaluation
│   ├── cli.py               # CLI commands (ingest/serve/eval)
│   └── vector_stores/
│       └── pinecone_store.py # Phase 5: Production Pinecone backend
├── tests/
│   ├── test_ingestion.py    # Unit tests for chunking strategies
│   ├── test_retrieval.py    # Unit tests for RRF + hybrid retriever
│   └── test_api.py          # FastAPI endpoint tests
├── data/
│   └── eval_golden_dataset.json  # RAGAS evaluation dataset
├── Dockerfile               # Multi-stage build (uv + slim-bookworm)
├── docker-compose.yml       # API + MLflow + Nginx services
├── nginx.conf               # SSE-optimised reverse proxy config
├── pyproject.toml           # uv project configuration
└── .env.example             # Environment variable template
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `CHUNK_STRATEGY` | `late_chunking` | `late_chunking \| parent_child \| recursive` |
| `EMBEDDING_MODEL` | `jinaai/jina-embeddings-v2-base-en` | Long-context embedding model |
| `VECTOR_STORE_BACKEND` | `faiss` | `faiss \| pinecone` |
| `BM25_WEIGHT` | `0.3` | Sparse retrieval weight (0.0–1.0) |
| `DENSE_WEIGHT` | `0.7` | Dense retrieval weight (0.0–1.0) |
| `INITIAL_RETRIEVAL_K` | `40` | Candidates before re-ranking |
| `FINAL_RERANK_K` | `5` | Documents passed to LLM |
| `RERANKER_BACKEND` | `flashrank` | `flashrank \| bge \| cohere` |
| `OPENAI_MODEL` | `gpt-4o-mini` | Generation model |

---

## RAGAS Metric Targets

| Metric | Target | Formula |
|---|---|---|
| **Faithfulness** | ≥ 0.90 | `\|S\| / \|C\|` — supported claims / all claims |
| **Context Precision** | ≥ 0.80 | Precision@K weighted by relevance indicators |
| **Answer Relevancy** | ≥ 0.80 | `cos(E_generated_q, E_original_q)` mean |
| **Context Recall** | ≥ 0.75 | Ground-truth statements found in context |

---

## AI Engineering Notes

Short, source-backed notes that document the production RAG and agentic design decisions behind this project:

- [Retrieval Quality Gates](docs/ai-insights/2026-07-15-retrieval-quality-gates.md)
- [Action Contracts for Agentic RAG](docs/ai-insights/2026-07-17-agentic-rag-action-contracts.md)
- [RAG Evaluation Drift](docs/ai-insights/2026-07-18-rag-evaluation-drift.md)
- [MCP Tool Boundaries for Evidence-Aware Agents](docs/ai-insights/2026-07-19-mcp-tool-boundaries.md)
- [Trace-to-Eval Loop for RAG](docs/ai-insights/2026-07-21-trace-to-eval-loop.md)
- [RAG Evidence Tracing: Debug the Answer Back to the Chunk](docs/ai-insights/2026-07-22-rag-evidence-tracing.md)

---

## Phase 5: Migrating to Pinecone

```bash
# Set in .env:
VECTOR_STORE_BACKEND=pinecone
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=docuquery-prod

# Re-run ingestion — it will auto-create the Pinecone index and upsert all chunks:
docuquery ingest --source ./data/documents
```

---

## License

MIT © DocuQuery Team
