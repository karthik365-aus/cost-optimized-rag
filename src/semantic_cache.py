"""
Semantic cache for near-duplicate queries.
"""
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.shared_embeddings import get_embedding_model

LOG = logging.getLogger(__name__)

class SemanticCache:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", max_entries: int = 500):
        self.model_name = model_name
        self.max_entries = int(max_entries)
        self.entries: List[Dict[str, Any]] = []

    def _get_embedding_model(self):
        """Reuse shared in-process embedding cache."""
        return get_embedding_model(self.model_name)

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        model = self._get_embedding_model()
        if model is None:
            return None
        try:
            vec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(vec, dtype=float).reshape(1, -1)
        except TypeError:
            vec = model.encode(query, convert_to_numpy=True)
            vec = np.asarray(vec, dtype=float)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            return vec.reshape(1, -1)
        except Exception:
            return None

    def lookup(self, query: str, corpus_hash: str) -> Optional[Dict[str, Any]]:
        query_emb = self._embed_query(query)
        if query_emb is None:
            return None

        candidates = [e for e in self.entries if e.get("corpus_hash", "") == (corpus_hash or "")]
        if not candidates:
            return None

        emb_matrix = np.vstack([np.asarray(e["embedding"], dtype=float).reshape(1, -1) for e in candidates])
        sims = cosine_similarity(query_emb, emb_matrix).flatten()
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim < 0.92:
            return None

        cached_result = dict(candidates[best_idx]["result"])
        cached_result["cache_hit"] = True
        cached_result["cache_similarity"] = round(best_sim, 4)
        return cached_result

    def store(self, query: str, result: Dict[str, Any], corpus_hash: str) -> None:
        if not isinstance(result, dict):
            LOG.debug("[SemanticCache] store skipped: result is not a dict")
            return
        if float(result.get("confidence_score_final", 0.0)) <= 0.75:
            LOG.debug(
                "[SemanticCache] store skipped: confidence_score_final=%s <= 0.75",
                result.get("confidence_score_final", 0.0),
            )
            return
        if bool(result.get("retried", False)):
            LOG.debug("[SemanticCache] store skipped: retried=True")
            return
        if float(result.get("coverage_score", 0.0)) <= 0.55:
            LOG.debug(
                "[SemanticCache] store skipped: coverage_score=%s <= 0.55",
                result.get("coverage_score", 0.0),
            )
            return
        if bool(result.get("retrieval_retry", False)):
            LOG.debug("[SemanticCache] store skipped: retrieval_retry=True")
            return

        query_emb = self._embed_query(query)
        if query_emb is None:
            LOG.debug("[SemanticCache] store skipped: query embedding unavailable")
            return

        self.entries.append(
            {
                "query": query,
                "embedding": query_emb.flatten(),
                "result": dict(result),
                "corpus_hash": corpus_hash or "",
            }
        )
        LOG.debug("[SemanticCache] stored entry. cache_size=%s", len(self.entries))
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
            LOG.debug("[SemanticCache] evicted oldest entry. cache_size=%s", len(self.entries))

    @staticmethod
    def get_corpus_hash(chroma_dir: str) -> str:
        if not chroma_dir or not os.path.isdir(chroma_dir):
            return ""
        items: List[str] = []
        for root, _, files in os.walk(chroma_dir):
            for filename in sorted(files):
                fp = os.path.join(root, filename)
                try:
                    mtime = os.path.getmtime(fp)
                except OSError:
                    continue
                items.append(f"{fp}:{mtime}")
        digest = hashlib.md5()
        digest.update("|".join(items).encode("utf-8"))
        return digest.hexdigest()
