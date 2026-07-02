# =============================================================================
# DocuQuery RAG — Multi-Stage Dockerfile
# Phase 5: Production containerisation with uv
# =============================================================================
# Target image size: < 300 MB
# Security: non-root user, no build tools in runtime stage
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder
# Uses the official uv image for ultra-fast dependency resolution (10-100x pip).
# ─────────────────────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Key uv performance flags:
#   UV_COMPILE_BYTECODE — pre-compile .py → .pyc, eliminates cold-start overhead
#   UV_LINK_MODE=copy   — required for cross-stage COPY (hardlinks don't survive)
#   UV_NO_DEV=1         — strip dev dependencies (pytest, ruff, mypy) from the image
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=never

# Install production dependencies first (cache layer — only invalidated if pyproject.toml changes)
# --mount=type=cache prevents re-downloading packages across rebuilds
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# Copy application source and install it into the venv
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime
# Minimal python:3.12-slim-bookworm — glibc compatible, no Alpine musl issues.
# Completely free of: uv binary, compilers, .c files, test fixtures.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Copy only the virtual environment produced by the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY --from=builder /app/app ./app
COPY --from=builder /app/pyproject.toml ./pyproject.toml

# Activate the virtual environment by prepending it to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Create persistent data directories
RUN mkdir -p /app/data/faiss_index /app/data/documents

# ─────────────────────────────────────────────────────────────────────────────
# Security: Run as non-root user (enterprise compliance requirement)
# ─────────────────────────────────────────────────────────────────────────────
RUN addgroup --system --gid 1001 docuquery && \
    adduser --system --uid 1001 --gid 1001 --no-create-home docuquery && \
    chown -R docuquery:docuquery /app

USER docuquery

# Expose the FastAPI port
EXPOSE 8000

# Health check for Kubernetes / ECS liveness probes
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/v1/health').raise_for_status()"

# Entrypoint: run uvicorn directly (no shell wrapper)
ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app", \
            "--host", "0.0.0.0", "--port", "8000", \
            "--log-config", "/dev/null"]
