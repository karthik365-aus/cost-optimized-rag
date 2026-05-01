"""
Context compression: turn retrieved chunks into a short string for the LLM.

Pipeline: extract text → sentences → score (TF-IDF + optional local sentence embeddings,
hybrid weight by ``complexity_score``) → light answer-type boosts → greedy pick with
redundancy pruning. Sentence budget: adaptive from ``complexity_score`` (or fixed map),
then **optional boost when retrieval keyword ``coverage_score`` is low** (more sentences
kept when retrieved chunks overlap the query poorly). Relevant env: ``COMPRESSION_*``.
"""
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.shared_embeddings import get_embedding_model


class _WhitespaceTokenizer:
    def encode(self, text: str):
        return text.split()


def _build_tokenizer():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return _WhitespaceTokenizer()


_TOKENIZER = _build_tokenizer()

_LABEL_SCORE_FALLBACK = {"simple": 0.2, "medium": 0.55, "complex": 0.85}


class ContextCompressor:
    """
    Score-first compression: hybrid TF-IDF + local embeddings (optional),
    adaptive sentence budget, redundancy pruning, light answer-type boosts.
    """

    def __init__(self, max_sentences_map=None, min_sentence_length: int = 20):
        self.max_sentences_map = max_sentences_map or {
            "simple": 2,
            "medium": 4,
            "complex": 6,
        }
        self.min_sentence_length = min_sentence_length
        self.use_embeddings = os.getenv("COMPRESSION_USE_EMBEDDINGS", "true").lower() == "true"
        self.embedding_model_name = os.getenv(
            "COMPRESSION_EMBEDDING_MODEL",
            "BAAI/bge-small-en-v1.5",
        )
        self.redundancy_threshold = float(os.getenv("COMPRESSION_REDUNDANCY_THRESHOLD", "0.85"))
        self.use_adaptive_budget = os.getenv("COMPRESSION_ADAPTIVE_BUDGET", "true").lower() == "true"
        loc_env = os.getenv("COMPRESSION_LOCATION_KEYWORDS", "").strip()
        if loc_env:
            self.location_keywords = tuple(
                k.strip().lower() for k in loc_env.split(",") if k.strip()
            )
        else:
            # Generic location cues (avoid corpus-specific geography names).
            self.location_keywords = (
                "campus", "building", "hall", "located", "location", "city", "state",
                "country", "region", "street", "avenue", "district", "near", "beside",
                "north", "south", "east", "west", "in front of",
            )
        self._embedding_model = None
        self._embedding_failed = False

    def compress(
        self,
        query: str,
        retrieved_docs: List[Any],
        complexity: str = "medium",
        complexity_score: Optional[float] = None,
        coverage_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        texts = self._extract_texts(retrieved_docs)
        original_text = "\n\n".join(texts).strip()
        normalized_complexity = (complexity or "medium").strip().lower()
        score_used = self._resolve_complexity_score(normalized_complexity, complexity_score)
        budget_meta = self._budget_from_complexity_and_coverage(
            normalized_complexity, score_used, coverage_score
        )
        budget = budget_meta["adaptive_top_n"]

        if not original_text:
            return self._empty_result(original_text, budget_meta)

        # --- Sentence prep ---
        sentences = self._split_into_sentences(original_text)
        sentences = self._clean_and_filter_sentences(sentences)

        if not sentences:
            original_tokens = self._count_tokens(original_text)
            return {
                "compressed_context": original_text,
                "selected_sentences": [],
                "all_sentences": [],
                "sentence_scores": [],
                "selected_indices": [],
                "original_text": original_text,
                "original_token_count": original_tokens,
                "compressed_token_count": original_tokens,
                "compression_ratio": 0.0,
                "compression_metadata": {
                    "complexity_score_used": round(float(score_used), 4),
                    "adaptive_top_n": 0,
                    "used_embeddings": False,
                    **{k: v for k, v in budget_meta.items() if k != "adaptive_top_n"},
                },
            }

        # --- Score sentences (hybrid) + answer-type boosts ---
        tfidf_scores = self._tfidf_scores(query, sentences)
        emb_scores, emb_matrix, used_embeddings = self._embedding_scores(query, sentences)

        if used_embeddings:
            w_tfidf, w_emb = self._hybrid_weights(score_used)
        else:
            w_tfidf, w_emb = 1.0, 0.0
        combined = w_tfidf * tfidf_scores + w_emb * emb_scores

        boosts = np.array([self._answer_boost(query, s) for s in sentences], dtype=float)
        final_scores = np.clip(combined + boosts, 0.0, 1.0)

        order = np.argsort(-final_scores)
        budget = min(budget, len(sentences))

        # --- Select top sentences; skip near-duplicates ---
        selected_idx, redundancy_skips = self._greedy_non_redundant(
            order=order,
            sentences=sentences,
            final_scores=final_scores,
            emb_matrix=emb_matrix,
            budget=budget,
        )
        selected_sentences = [sentences[i] for i in selected_idx]
        ranked_for_metadata = list(
            zip(
                sentences,
                tfidf_scores.tolist(),
                emb_scores.tolist(),
                boosts.tolist(),
                final_scores.tolist(),
            )
        )
        ranked_for_metadata.sort(key=lambda x: -x[4])

        sentence_scores = [
            {
                "sentence": s,
                "tfidf_score": round(float(t), 4),
                "embedding_score": round(float(e), 4),
                "answer_boost": round(float(b), 4),
                "final_score": round(float(f), 4),
            }
            for s, t, e, b, f in ranked_for_metadata
        ]

        compressed_context = " ".join(selected_sentences).strip()
        original_tokens = self._count_tokens(original_text)
        compressed_tokens = self._count_tokens(compressed_context)
        compression_ratio = (
            0.0 if original_tokens == 0 else (original_tokens - compressed_tokens) / original_tokens
        )

        return {
            "compressed_context": compressed_context,
            "selected_sentences": selected_sentences,
            "all_sentences": sentences,
            "sentence_scores": sentence_scores,
            "selected_indices": selected_idx,
            "original_text": original_text,
            "original_token_count": original_tokens,
            "compressed_token_count": compressed_tokens,
            "compression_ratio": round(compression_ratio, 4),
            "compression_metadata": {
                "complexity_score_used": round(float(score_used), 4),
                "adaptive_top_n": budget,
                "hybrid_weights": {"tfidf": w_tfidf, "embedding": w_emb},
                "used_embeddings": used_embeddings,
                "redundancy_threshold": self.redundancy_threshold,
                "redundancy_skips": redundancy_skips,
                **{k: v for k, v in budget_meta.items() if k != "adaptive_top_n"},
            },
        }

    def _empty_result(self, original_text: str, budget_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = dict(budget_meta or {})
        meta.setdefault("coverage_score_used", None)
        meta.setdefault("sentence_budget_base", None)
        meta.setdefault("coverage_budget_extra", 0)
        return {
            "compressed_context": "",
            "selected_sentences": [],
            "all_sentences": [],
            "sentence_scores": [],
            "selected_indices": [],
            "original_text": original_text,
            "original_token_count": 0,
            "compressed_token_count": 0,
            "compression_ratio": 0.0,
            "compression_metadata": meta,
        }

    def _resolve_complexity_score(self, label: str, complexity_score: Optional[float]) -> float:
        if complexity_score is not None:
            return max(0.0, min(1.0, float(complexity_score)))
        return _LABEL_SCORE_FALLBACK.get(label, 0.55)

    def _sentence_budget(self, label: str, complexity_score: float) -> int:
        if self.use_adaptive_budget:
            return max(2, min(6, round(2 + float(complexity_score) * 4)))
        return self.max_sentences_map.get(label, 4)

    def _budget_from_complexity_and_coverage(
        self,
        label: str,
        complexity_score: float,
        coverage_score: Optional[float],
    ) -> Dict[str, Any]:
        """
        Base budget from analyzer complexity_score; if retrieval coverage is low,
        add up to +2 sentences (capped at 6) so weak keyword overlap keeps more context.
        """
        base = self._sentence_budget(label, complexity_score)
        if coverage_score is None:
            return {
                "sentence_budget_base": base,
                "coverage_score_used": None,
                "coverage_budget_extra": 0,
                "adaptive_top_n": base,
            }
        cov = max(0.0, min(1.0, float(coverage_score)))
        extra = int(round((1.0 - cov) * 2))
        final = max(2, min(6, base + extra))
        return {
            "sentence_budget_base": base,
            "coverage_score_used": round(cov, 4),
            "coverage_budget_extra": extra,
            "adaptive_top_n": final,
        }

    def _hybrid_weights(self, complexity_score: float) -> Tuple[float, float]:
        """Increase embedding weight as complexity rises (simple keyword-heavy → complex semantic-heavy)."""
        w_emb = 0.25 + 0.5 * float(complexity_score)
        w_emb = max(0.25, min(0.75, w_emb))
        w_tfidf = 1.0 - w_emb
        return w_tfidf, w_emb

    def _tfidf_scores(self, query: str, sentences: List[str]) -> np.ndarray:
        if len(sentences) == 1:
            return np.array([1.0], dtype=float)
        corpus = [query] + sentences
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(corpus)
            sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        except Exception:
            sims = np.ones(len(sentences), dtype=float) / len(sentences)
        return self._minmax01(sims)

    def _embedding_scores(
        self, query: str, sentences: List[str]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        n = len(sentences)
        if not self.use_embeddings or self._embedding_failed:
            return np.ones(n, dtype=float) / max(n, 1), None, False

        model = self._get_embedding_model()
        if model is None:
            return np.ones(n, dtype=float) / max(n, 1), None, False

        try:
            texts = [query] + sentences
            try:
                emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            except TypeError:
                emb = model.encode(texts, convert_to_numpy=True)
                emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            qv = emb[0:1]
            sv = emb[1:]
            sims = cosine_similarity(qv, sv).flatten()
            return self._minmax01(sims), sv, True
        except Exception:
            return np.ones(n, dtype=float) / max(n, 1), None, False

    def _get_embedding_model(self):
        if self._embedding_failed:
            return None
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            self._embedding_model = get_embedding_model(self.embedding_model_name)
            if self._embedding_model is None:
                self._embedding_failed = True
            return self._embedding_model
        except Exception:
            self._embedding_failed = True
            return None

    @staticmethod
    def _minmax01(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-9:
            return np.ones_like(arr) * 0.5
        return (arr - lo) / (hi - lo)

    def _answer_boost(self, query: str, sentence: str) -> float:
        q = (query or "").lower().strip()
        s = sentence or ""
        s_low = s.lower()
        boost = 0.0

        if re.search(r"\bwhen\b|what year|which year|date of", q):
            if re.search(r"\b(19|20)\d{2}\b", s):
                boost += 0.08

        if re.match(r"who\b", q) or " who " in f" {q} ":
            caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", s)
            if len(caps) >= 1:
                boost += 0.06
        if re.match(r"where\b", q) or " where " in f" {q} ":
            # Generic location cues + proper-noun patterns.
            has_loc_kw = any(k in s_low for k in self.location_keywords)
            has_place_pattern = bool(
                re.search(r"\b(in|at|near|inside|outside)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", s)
            )
            if has_loc_kw or has_place_pattern:
                boost += 0.06

        if q.startswith("what is") or q.startswith("what was") or "what is the" in q:
            if re.search(r"\bis a\b|\bis an\b|\brefers to\b|\bmeans\b|defined as", s, re.I):
                boost += 0.05

        return min(boost, 0.15)

    def _greedy_non_redundant(
        self,
        order: np.ndarray,
        sentences: List[str],
        final_scores: np.ndarray,
        emb_matrix: Optional[np.ndarray],
        budget: int,
    ) -> Tuple[List[int], int]:
        selected: List[int] = []
        skips = 0
        tfidf_sparse = None
        redundancy_matrix = emb_matrix

        if redundancy_matrix is None and len(sentences) > 1:
            try:
                vectorizer = TfidfVectorizer(stop_words="english")
                tfidf_sparse = vectorizer.fit_transform(sentences)
            except Exception:
                tfidf_sparse = None

        for idx in order:
            if len(selected) >= budget:
                break
            if redundancy_matrix is not None:
                redundant = False
                vec_i = redundancy_matrix[idx]
                for j in selected:
                    vec_j = redundancy_matrix[j]
                    num = float(np.dot(vec_i, vec_j))
                    den = float(np.linalg.norm(vec_i) * np.linalg.norm(vec_j)) or 1.0
                    sim = num / den
                    if sim >= self.redundancy_threshold:
                        redundant = True
                        break
                if redundant:
                    skips += 1
                    continue
            elif tfidf_sparse is not None:
                redundant = False
                for j in selected:
                    sim = float(
                        cosine_similarity(
                            tfidf_sparse[idx : idx + 1],
                            tfidf_sparse[j : j + 1],
                        )[0, 0]
                    )
                    if sim >= self.redundancy_threshold:
                        redundant = True
                        break
                if redundant:
                    skips += 1
                    continue
            selected.append(int(idx))

        if not selected:
            best = int(np.argmax(final_scores))
            selected = [best]

        return selected, skips

    def _extract_texts(self, retrieved_docs: List[Any]) -> List[str]:
        texts = []
        for doc in retrieved_docs:
            if hasattr(doc, "page_content"):
                texts.append(doc.page_content)
            elif isinstance(doc, str):
                texts.append(doc)
        return texts

    def _split_into_sentences(self, text: str) -> List[str]:
        text = text.replace("\n", " ").strip()
        if not text:
            return []
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _clean_and_filter_sentences(self, sentences: List[str]) -> List[str]:
        seen = set()
        cleaned = []

        for sentence in sentences:
            normalized = re.sub(r"\s+", " ", sentence).strip()
            key = normalized.lower()

            if len(normalized) < self.min_sentence_length:
                continue
            if key in seen:
                continue

            seen.add(key)
            cleaned.append(normalized)

        return cleaned

    def _count_tokens(self, text: str) -> int:
        return len(_TOKENIZER.encode(text))
