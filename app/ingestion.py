"""
Phase 1 — Ingestion & Storage
==============================
Three chunking strategies are implemented as interchangeable strategies:

1. LateChunkingProcessor   — token-level embeddings from long-context model,
                             chunk boundaries applied post-encoding.
2. ParentChildProcessor    — LangChain ParentDocumentRetriever pattern.
3. RecursiveProcessor      — Baseline recursive character splitting.

All processors expose the same interface:
    process(documents) -> list[Document]

The ingestion pipeline ties together loading, chunking, embedding, and
indexing into FAISS (dev) or Pinecone (prod).
"""

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from app.config import ChunkStrategy, Settings, get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


class JinaEmbedder:
    """
    Wraps jinaai/jina-embeddings-v2-base-en (8 192-token context window).
    Used for both Late Chunking and standard dense retrieval.
    """

    def __init__(self, model_name: str, max_seq_len: int = 8192) -> None:
        from transformers import AutoModel, AutoTokenizer

        logger.info("loading_jina_model", model=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.max_seq_len = max_seq_len

    def _mean_pool(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Mean pooling over non-padding tokens."""
        mask = attention_mask[:, :, np.newaxis].astype(float)
        summed = (token_embeddings * mask).sum(axis=1)
        counts = mask.sum(axis=1).clip(min=1e-9)
        return summed / counts

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Standard dense embedding — one vector per text."""
        import torch

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**encoded)

        embeddings = self._mean_pool(
            outputs.last_hidden_state.numpy(),
            encoded["attention_mask"].numpy(),
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def late_chunk_embed(
        self,
        text: str,
        chunk_spans: list[tuple[int, int]],  # character-level spans
    ) -> list[list[float]]:
        """
        Late Chunking implementation:
          1. Tokenize the *full* document (up to max_seq_len).
          2. Forward pass — get token-level embeddings with global attention.
          3. Map character spans → token spans.
          4. Mean-pool each token span to produce one vector per chunk.
        """
        import torch

        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        offset_mapping = encoded.pop("offset_mapping")[0].numpy()  # (seq_len, 2)

        with torch.no_grad():
            outputs = self.model(**encoded)

        token_embeddings = outputs.last_hidden_state[0].numpy()  # (seq_len, hidden)

        chunk_vectors: list[list[float]] = []
        for char_start, char_end in chunk_spans:
            # Find token indices whose offsets overlap with this char span
            token_mask = (offset_mapping[:, 0] < char_end) & (offset_mapping[:, 1] > char_start)
            selected = token_embeddings[token_mask]
            if len(selected) == 0:
                # Fallback: embed the chunk text in isolation
                vec = self.embed_documents([text[char_start:char_end]])[0]
            else:
                vec = selected.mean(axis=0).tolist()
            chunk_vectors.append(vec)

        return chunk_vectors


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------


class BaseChunkingProcessor(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def process(self, documents: list[Document]) -> list[Document]:
        """
        Split source documents into indexable chunks.

        Returns a list of Document objects where each doc represents
        one chunk ready for embedding.
        """

    @staticmethod
    def _ensure_metadata(chunk: Document, source_doc: Document, idx: int) -> None:
        chunk.metadata.setdefault("source", source_doc.metadata.get("source", "unknown"))
        chunk.metadata.setdefault("chunk_index", idx)


class RecursiveProcessor(BaseChunkingProcessor):
    """Baseline recursive character splitter (Phase 1 fallback)."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in documents:
            splits = self.splitter.split_documents([doc])
            for i, chunk in enumerate(splits):
                self._ensure_metadata(chunk, doc, i)
            chunks.extend(splits)
        logger.info("recursive_chunking_done", num_chunks=len(chunks))
        return chunks


class ParentChildProcessor(BaseChunkingProcessor):
    """
    Hierarchical chunking: large parent chunks stored in a docstore,
    small child chunks embedded in the vector index.
    Uses the same IDs so ParentDocumentRetriever can look up parents.
    """

    def __init__(self, parent_size: int = 2000, child_size: int = 400) -> None:
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size, chunk_overlap=200
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size, chunk_overlap=50
        )

    def process(self, documents: list[Document]) -> list[Document]:
        """
        Returns child chunks with `parent_id` metadata set.
        Callers should also persist parent chunks separately.
        """
        all_children: list[Document] = []
        for doc in documents:
            parents = self.parent_splitter.split_documents([doc])
            for p_idx, parent in enumerate(parents):
                parent_id = f"{doc.metadata.get('source', 'doc')}::parent::{p_idx}"
                parent.metadata["parent_id"] = parent_id
                children = self.child_splitter.split_documents([parent])
                for c_idx, child in enumerate(children):
                    child.metadata["parent_id"] = parent_id
                    self._ensure_metadata(child, doc, c_idx)
                all_children.extend(children)

        logger.info("parent_child_chunking_done", num_children=len(all_children))
        return all_children


class LateChunkingProcessor(BaseChunkingProcessor):
    """
    Late Chunking strategy:
    1. Sentence-boundary splitting to derive character-level chunk spans.
    2. Full document passed to JinaEmbedder.late_chunk_embed().
    3. Returns Documents whose embeddings encode global document context.
    """

    def __init__(self, embedder: JinaEmbedder, chunk_size: int = 400) -> None:
        self.embedder = embedder
        self.chunk_size = chunk_size
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _get_char_spans(self, text: str) -> list[tuple[int, int]]:
        """Derive character-level (start, end) spans from sentence splitting."""
        chunks = self._splitter.split_text(text)
        spans: list[tuple[int, int]] = []
        cursor = 0
        for chunk in chunks:
            start = text.find(chunk, cursor)
            if start == -1:
                start = cursor
            end = start + len(chunk)
            spans.append((start, end))
            cursor = end
        return spans

    def process(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            text = doc.page_content
            if not text.strip():
                continue

            spans = self._get_char_spans(text)
            if not spans:
                continue

            try:
                vectors = self.embedder.late_chunk_embed(text, spans)
            except Exception as exc:
                logger.warning("late_chunk_embed_failed", error=str(exc), source=doc.metadata.get("source"))
                # Graceful fallback: emit raw text chunks without precomputed vectors
                fallback = self._splitter.split_documents([doc])
                result.extend(fallback)
                continue

            for idx, (span, vector) in enumerate(zip(spans, vectors, strict=False)):
                chunk_text = text[span[0] : span[1]]
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "char_start": span[0],
                        "char_end": span[1],
                        "precomputed_embedding": vector,  # stored for direct FAISS insertion
                        "chunking_strategy": "late_chunking",
                    },
                )
                result.append(chunk_doc)

        logger.info("late_chunking_done", num_chunks=len(result))
        return result


# ---------------------------------------------------------------------------
# Document Loader
# ---------------------------------------------------------------------------


class DocumentLoader:
    """Unified loader for PDF, DOCX, TXT, and Markdown files."""

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".docx": UnstructuredWordDocumentLoader,
    }

    def load_directory(self, directory: str | Path) -> list[Document]:
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        docs: list[Document] = []
        for ext, loader_cls in self.LOADER_MAP.items():
            pattern = f"**/*{ext}"
            try:
                loader = DirectoryLoader(
                    str(directory),
                    glob=pattern,
                    loader_cls=loader_cls,
                    show_progress=True,
                    use_multithreading=True,
                )
                loaded = loader.load()
                logger.info("loaded_files", ext=ext, count=len(loaded))
                docs.extend(loaded)
            except Exception as exc:
                logger.warning("loader_error", ext=ext, error=str(exc))

        logger.info("total_documents_loaded", count=len(docs))
        return docs

    def load_file(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)
        ext = path.suffix.lower()
        loader_cls = self.LOADER_MAP.get(ext, TextLoader)
        loader = loader_cls(str(path))
        return loader.load()


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------


class BM25Index:
    """
    Wraps rank-bm25 with persistence (pickle) and a LangChain-compatible
    retriever interface.
    """

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self._documents: list[Document] = []

    def build(self, documents: list[Document]) -> None:
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self._index = BM25Okapi(tokenized)
        self._documents = documents
        logger.info("bm25_index_built", num_docs=len(documents))

    def search(self, query: str, k: int = 20) -> list[Document]:
        if self._index is None:
            raise RuntimeError("BM25 index not built. Call build() first.")
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)
        top_k = np.argsort(scores)[::-1][:k]
        return [self._documents[i] for i in top_k if scores[i] > 0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"index": self._index, "documents": self._documents}, f)
        logger.info("bm25_index_saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance._index = data["index"]
        instance._documents = data["documents"]
        logger.info("bm25_index_loaded", path=str(path), num_docs=len(instance._documents))
        return instance


# ---------------------------------------------------------------------------
# FAISS Vector Store
# ---------------------------------------------------------------------------


class FAISSVectorStore:
    """
    FAISS-backed dense vector store with manual persistence.
    Supports both standard embeddings and precomputed Late Chunking vectors.
    """

    def __init__(self, dimension: int = 768) -> None:
        import faiss

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalisation)
        self._documents: list[Document] = []

    def add_documents(
        self,
        documents: list[Document],
        embedder: JinaEmbedder | None = None,
    ) -> None:
        import faiss

        vectors: list[list[float]] = []
        docs_to_add: list[Document] = []

        for doc in documents:
            precomputed = doc.metadata.get("precomputed_embedding")
            if precomputed is not None:
                vectors.append(precomputed)
                docs_to_add.append(doc)
            elif embedder is not None:
                vectors.append(embedder.embed_documents([doc.page_content])[0])
                docs_to_add.append(doc)
            else:
                logger.warning("skipping_doc_no_embedding", source=doc.metadata.get("source"))

        if not vectors:
            return

        arr = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(arr)
        self.index.add(arr)
        self._documents.extend(docs_to_add)
        logger.info("faiss_docs_added", total=len(self._documents))

    def search(self, query_vector: list[float], k: int = 20) -> list[tuple[Document, float]]:
        import faiss

        arr = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(arr)
        distances, indices = self.index.search(arr, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self._documents):
                results.append((self._documents[idx], float(dist)))
        return results

    def save(self, directory: str | Path) -> None:
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)
        logger.info("faiss_index_saved", path=str(directory))

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSVectorStore":
        import faiss

        directory = Path(directory)
        index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "documents.pkl", "rb") as f:
            documents = pickle.load(f)
        instance = cls(dimension=index.d)
        instance.index = index
        instance._documents = documents
        logger.info("faiss_index_loaded", path=str(directory), num_docs=len(documents))
        return instance


# ---------------------------------------------------------------------------
# Ingestion Pipeline (orchestrator)
# ---------------------------------------------------------------------------


class IngestionPipeline:
    """
    End-to-end ingestion orchestrator.
    Loads → Chunks → Embeds → Indexes (FAISS + BM25).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._embedder: JinaEmbedder | None = None
        self._faiss: FAISSVectorStore | None = None
        self._bm25: BM25Index | None = None

    @property
    def embedder(self) -> JinaEmbedder:
        if self._embedder is None:
            self._embedder = JinaEmbedder(
                model_name=self.settings.embedding_model,
                max_seq_len=self.settings.embedding_max_seq_len,
            )
        return self._embedder

    def _build_processor(self) -> BaseChunkingProcessor:
        strategy = self.settings.chunk_strategy
        if strategy == ChunkStrategy.LATE_CHUNKING:
            return LateChunkingProcessor(
                embedder=self.embedder,
                chunk_size=self.settings.child_chunk_size,
            )
        elif strategy == ChunkStrategy.PARENT_CHILD:
            return ParentChildProcessor(
                parent_size=self.settings.parent_chunk_size,
                child_size=self.settings.child_chunk_size,
            )
        else:
            return RecursiveProcessor(chunk_size=self.settings.child_chunk_size)

    def run(self, source_directory: str | Path) -> dict[str, Any]:
        """
        Full ingestion run.
        Returns a summary dict with counts and artifact paths.
        """
        loader = DocumentLoader()
        raw_docs = loader.load_directory(source_directory)
        if not raw_docs:
            raise ValueError(f"No documents found in {source_directory}")

        processor = self._build_processor()
        chunks = processor.process(raw_docs)

        # ── FAISS ───────────────────────────────────────────────────────────
        self._faiss = FAISSVectorStore(dimension=768)
        self._faiss.add_documents(
            chunks,
            embedder=self.embedder if self.settings.chunk_strategy != ChunkStrategy.LATE_CHUNKING else None,
        )
        faiss_path = Path(self.settings.faiss_index_path)
        self._faiss.save(faiss_path)

        # ── BM25 ────────────────────────────────────────────────────────────
        self._bm25 = BM25Index()
        self._bm25.build(chunks)
        bm25_path = faiss_path / "bm25_index.pkl"
        self._bm25.save(bm25_path)

        summary = {
            "raw_documents": len(raw_docs),
            "chunks": len(chunks),
            "strategy": self.settings.chunk_strategy.value,
            "faiss_path": str(faiss_path),
            "bm25_path": str(bm25_path),
        }
        logger.info("ingestion_complete", **summary)
        return summary
