"""
Centralised application configuration using Pydantic-Settings.
All values are loaded from environment variables / .env file.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorBackend(str, Enum):
    FAISS = "faiss"
    PINECONE = "pinecone"


class ChunkStrategy(str, Enum):
    LATE_CHUNKING = "late_chunking"
    PARENT_CHILD = "parent_child"
    RECURSIVE = "recursive"


class RerankerBackend(str, Enum):
    FLASHRANK = "flashrank"
    BGE = "bge"
    COHERE = "cohere"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # ── Pinecone ─────────────────────────────────────────────────────────────
    pinecone_api_key: str = Field(default="")
    pinecone_environment: str = Field(default="us-east-1-aws")
    pinecone_index_name: str = Field(default="docuquery-prod")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="docuquery-rag-v1")

    # ── Application ───────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = Field(default="INFO")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://localhost:8000"])
    api_secret_key: str = Field(default="change-me-in-production")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    vector_store_backend: VectorBackend = VectorBackend.FAISS
    faiss_index_path: str = Field(default="./data/faiss_index")
    chunk_strategy: ChunkStrategy = ChunkStrategy.LATE_CHUNKING
    parent_chunk_size: int = Field(default=2000, ge=512)
    child_chunk_size: int = Field(default=400, ge=64)
    embedding_model: str = Field(default="jinaai/jina-embeddings-v2-base-en")
    embedding_max_seq_len: int = Field(default=8192)

    # ── Hybrid Retrieval ──────────────────────────────────────────────────────
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    initial_retrieval_k: int = Field(default=40, ge=5)
    final_rerank_k: int = Field(default=5, ge=1)

    # ── Re-ranking ────────────────────────────────────────────────────────────
    reranker_backend: RerankerBackend = RerankerBackend.FLASHRANK
    cohere_api_key: str = Field(default="")

    # ── Streaming ─────────────────────────────────────────────────────────────
    stream_timeout_seconds: int = Field(default=60)
    max_context_tokens: int = Field(default=4096)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except Exception:
                return [v]
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def use_pinecone(self) -> bool:
        return self.vector_store_backend == VectorBackend.PINECONE


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
