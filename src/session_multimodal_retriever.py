"""
Session-only multimodal retrieval for uploaded visual PDFs.

This module is intentionally in-memory only:
- no persistence
- no writes to main chroma_db
- no impact on the main corpus index
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Dict, List

import chromadb
import numpy as np
from chromadb.api.types import EmbeddingFunction

from src.shared_embeddings import get_embedding_model

LOG = logging.getLogger(__name__)


class SessionMultimodalError(Exception):
    """Raised when session multimodal retrieval cannot be built or queried."""


def detect_visual_pdf(pdf_bytes: bytes) -> bool:
    """Return True when any page in the PDF has embedded images."""
    if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
        return False
    try:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            for page in doc:
                if len(page.get_images()) > 0:
                    return True
        finally:
            doc.close()
        return False
    except Exception:
        return False


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Small adapter so Chroma can consume Gemini embeddings."""
    MODEL_NAME = "models/gemini-embedding-2-preview"
    OUTPUT_DIM = 768

    def __init__(self, api_key: str):
        self.api_key = (api_key or "").strip()
        self._genai = None

    def _ensure_client(self):
        if self._genai is not None:
            return self._genai
        if not self.api_key:
            raise SessionMultimodalError("GEMINI_API_KEY is missing")
        try:
            import google.generativeai as genai
        except Exception as exc:
            raise SessionMultimodalError(f"google-generativeai unavailable: {exc}") from exc
        genai.configure(api_key=self.api_key)
        self._genai = genai
        return self._genai

    def _embed_one(self, text: str, task_type: str) -> List[float]:
        genai = self._ensure_client()
        response = genai.embed_content(
            model=self.MODEL_NAME,
            content=text,
            task_type=task_type,
            output_dimensionality=self.OUTPUT_DIM,
        )
        vector = response.get("embedding") if isinstance(response, dict) else getattr(response, "embedding", None)
        if vector is None:
            raise SessionMultimodalError("Gemini embedding response missing vector")
        vec = [float(x) for x in vector]
        if not vec:
            raise SessionMultimodalError("Gemini embedding returned empty vector")
        return vec

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(str(t or ""), task_type="retrieval_document") for t in texts]


class SessionMultimodalRetriever:
    """Build and query a session-scoped multimodal index for one uploaded PDF."""
    TOP_K = 3
    _CHROMA_CLIENT = None
    _SESSION_COLLECTIONS: Dict[str, object] = {}
    _SESSION_STATS: Dict[str, Dict[str, float]] = {}

    def __init__(self, api_key: str, pdf_bytes: bytes, pdf_text_by_page: dict, use_gemini: bool = True):
        self.api_key = (api_key or "").strip()
        self.use_gemini = bool(use_gemini)
        self.pdf_bytes = bytes(pdf_bytes or b"")
        self.session_id = hashlib.md5(self.pdf_bytes).hexdigest()[:12]
        self.collection_name = (
            f"session_{self.session_id}" if self.use_gemini else f"session_text_{self.session_id}"
        )
        self.pdf_text_by_page = {
            int(k): str(v)
            for k, v in (pdf_text_by_page or {}).items()
            if isinstance(v, str) and v.strip()
        }
        self._embedding_function = GeminiEmbeddingFunction(self.api_key) if self.use_gemini else None
        self._bge_model_name = "BAAI/bge-small-en-v1.5"
        self._bge_model = None
        self._collection = None
        self.chunk_count = 0
        self.avg_chunk_tokens = 0.0
        self.last_build_reused_cache = False

    @classmethod
    def _get_client(cls):
        if cls._CHROMA_CLIENT is None:
            cls._CHROMA_CLIENT = chromadb.Client()
        return cls._CHROMA_CLIENT

    def _get_or_create_collection(self):
        if self._collection is not None:
            return self._collection
        cached = self._SESSION_COLLECTIONS.get(self.collection_name)
        if cached is not None:
            self._collection = cached
            return cached
        client = self._get_client()
        # One collection per uploaded PDF hash; remains in-memory for this process only.
        metadata = (
            {
                "embedding_model": self._embedding_function.MODEL_NAME,
                "embedding_dim": self._embedding_function.OUTPUT_DIM,
            }
            if self.use_gemini
            else {
                "embedding_model": self._bge_model_name,
            }
        )
        self._collection = client.get_or_create_collection(name=self.collection_name, metadata=metadata)
        self._SESSION_COLLECTIONS[self.collection_name] = self._collection
        return self._collection

    @staticmethod
    def _filter_chunks(chunks: List[str]) -> List[str]:
        usable: List[str] = []
        for raw in chunks:
            chunk = (raw or "").strip()
            if len(chunk) < 30:
                continue
            if len(chunk) > 700:
                chunk = chunk[:700]
            usable.append(chunk)
        return usable

    def _chunks_for_page(self, page_text: str) -> List[str]:
        primary = re.split(r"(?<=[.!?])\s+", page_text or "")
        primary_usable = self._filter_chunks(primary)
        if len(primary_usable) >= 2:
            return primary_usable
        fallback = (page_text or "").split("\n\n")
        return self._filter_chunks(fallback)

    def _embed(self, text: str, task_type: str) -> List[float]:
        if self.use_gemini:
            return self._embedding_function._embed_one(text, task_type=task_type)  # type: ignore[union-attr]
        if self._bge_model is None:
            self._bge_model = get_embedding_model(self._bge_model_name)
            if self._bge_model is None:
                raise SessionMultimodalError(f"Unable to load embedding model: {self._bge_model_name}")
        vec = self._bge_model.encode(text, normalize_embeddings=True)
        return [float(x) for x in np.asarray(vec, dtype=float).reshape(-1)]

    def build_index(self) -> int:
        """Create (or reuse) the session collection and return indexed chunk count."""
        collection = self._get_or_create_collection()
        existing_count = int(collection.count() or 0)
        if existing_count > 0:
            # Cache hit: embeddings were already built for this session PDF.
            self.last_build_reused_cache = True
            self.chunk_count = existing_count
            stats = self._SESSION_STATS.get(self.collection_name, {})
            self.avg_chunk_tokens = float(stats.get("avg_chunk_tokens", 0.0))
            return existing_count

        candidate_chunks: List[tuple[int, int, str]] = []
        for page_num in sorted(self.pdf_text_by_page.keys()):
            page_text = self.pdf_text_by_page.get(page_num, "")
            chunks = self._chunks_for_page(page_text)
            for chunk_idx, chunk_text in enumerate(chunks):
                candidate_chunks.append((page_num, chunk_idx, chunk_text))

        if not candidate_chunks:
            raise SessionMultimodalError("No chunks produced after filtering")

        ids: List[str] = []
        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict] = []
        token_counts: List[int] = []

        for page_num, chunk_idx, chunk_text in candidate_chunks:
            try:
                vector = self._embed(chunk_text, task_type="retrieval_document")
            except Exception as exc:
                LOG.warning(
                    "[SessionMultimodalRetriever] chunk embedding failed (page=%s chunk=%s): %s",
                    page_num,
                    chunk_idx,
                    exc,
                )
                continue
            ids.append(f"{self.collection_name}_{page_num}_{chunk_idx}")
            documents.append(chunk_text)
            embeddings.append(vector)
            metadatas.append({"page_num": int(page_num), "chunk_idx": int(chunk_idx)})
            token_counts.append(len(chunk_text.split()))

        if not documents:
            raise SessionMultimodalError("All chunk embeddings failed")

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        self.last_build_reused_cache = False
        self.chunk_count = len(documents)
        self.avg_chunk_tokens = float(sum(token_counts) / max(1, len(token_counts)))
        self._SESSION_STATS[self.collection_name] = {
            "chunk_count": float(self.chunk_count),
            "avg_chunk_tokens": float(self.avg_chunk_tokens),
        }
        return self.chunk_count

    def retrieve(self, query: str) -> List[Dict]:
        """Return top multimodal chunks for a query from the session collection."""
        collection = self._get_or_create_collection()
        if int(collection.count() or 0) == 0:
            raise SessionMultimodalError("Session index is empty; call build_index() first")

        q = (query or "").strip()
        if not q:
            return []
        query_vec = np.asarray(self._embed(q, task_type="retrieval_query"), dtype=float).reshape(1, -1)

        result = collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=self.TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        out: List[Dict] = []
        for doc_text, meta, dist in zip(docs, metas, dists):
            # Chroma returns distance; convert to a similarity-like score for UI diagnostics.
            out.append(
                {
                    "text": str(doc_text or "").strip(),
                    "page_num": int((meta or {}).get("page_num", 0)),
                    "score": float(1.0 - float(dist)) if dist is not None else 0.0,
                }
            )
        return out
