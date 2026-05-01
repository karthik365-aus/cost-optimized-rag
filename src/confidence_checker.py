"""
Post-generation grounding check: scores the **answer against the compressed context**
(not against CSV ground truth). Combines heuristic overlap, TF-IDF, and optional
BGE sentence similarity; if aggregate confidence is below ``CONFIDENCE_THRESHOLD`` and
the model maps in ``MODEL_TIERS``, may retry once with a stronger OpenAI model.

``MODEL_TIERS`` keys are **OpenAI-style** model names (e.g. ``gpt-3.5-turbo``). If the
router used a **local** LM Studio id, the name will not match and **no tier retry** runs;
only scores are returned. Extend ``MODEL_TIERS`` deliberately if you need local retries.
"""
from typing import Dict, Any
import os
import re
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from src.shared_embeddings import get_embedding_model

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
_LEGACY_CONFIDENCE_THRESHOLD = os.getenv("CONFIDENCE_THRESHOLD")
if _LEGACY_CONFIDENCE_THRESHOLD is not None:
    # Backward-compatible single threshold override.
    CONFIDENCE_THRESHOLD_LOCAL = float(_LEGACY_CONFIDENCE_THRESHOLD)
    CONFIDENCE_THRESHOLD_OPENAI = float(_LEGACY_CONFIDENCE_THRESHOLD)
else:
    CONFIDENCE_THRESHOLD_LOCAL = float(os.getenv("CONFIDENCE_THRESHOLD_LOCAL", "0.50"))
    CONFIDENCE_THRESHOLD_OPENAI = float(os.getenv("CONFIDENCE_THRESHOLD_OPENAI", "0.65"))

_EMBEDDING_MODEL = None
_EMBEDDING_LOAD_ATTEMPTED = False
_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _get_embedding_model():
    """Lazy load so a missing ``sentence_transformers`` install does not break import."""
    global _EMBEDDING_MODEL, _EMBEDDING_LOAD_ATTEMPTED
    if _EMBEDDING_LOAD_ATTEMPTED:
        return _EMBEDDING_MODEL
    _EMBEDDING_LOAD_ATTEMPTED = True
    try:
        _EMBEDDING_MODEL = get_embedding_model(_EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"[ConfidenceChecker] Embedding model unavailable, continuing without it: {e}")
        _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL

def _build_model_tiers() -> Dict[str, str]:
    """
    Build retry/escalation map for both local and OpenAI models.

    Local router ids (from env) escalate to gpt-4o-mini by default, while
    OpenAI tiers keep the existing cascade.
    """
    tiers: Dict[str, str] = {
        "gpt-3.5-turbo": "gpt-4o-mini",
        "gpt-4o-mini": "gpt-4o",
    }
    local_simple = (os.getenv("LOCAL_SIMPLE_MODEL", "") or "").strip()
    local_medium = (os.getenv("LOCAL_MEDIUM_MODEL", "") or "").strip()
    local_targets = [local_simple, local_medium]
    for model_id in local_targets:
        if not model_id:
            continue
        tiers[model_id] = "gpt-4o-mini"
        # Also support short aliases (e.g., "ministral" from "mistralai/ministral-3-3b").
        short = model_id.split("/")[-1].strip()
        if short:
            tiers[short] = "gpt-4o-mini"
    return tiers


def _select_confidence_threshold(model_used: str, router_output: Dict[str, Any]) -> float:
    source = (router_output or {}).get("model_source", "")
    source = source.lower().strip()
    model_used = (model_used or "").lower().strip()
    if source == "openai" or model_used.startswith("gpt-"):
        return CONFIDENCE_THRESHOLD_OPENAI
    return CONFIDENCE_THRESHOLD_LOCAL

# -----------------------------
# HEURISTIC SCORING
# -----------------------------
def heuristic_score(answer: str, context: str, complexity: str = "medium") -> float:
    if not answer or len(answer.strip()) == 0:
        return 0.0

    answer_lower = answer.lower()
    bad_phrases = [
        "insufficient",
        "not mentioned",
        "cannot determine",
        "unclear",
        "not provided",
        "i could not find",
    ]
    if any(p in answer_lower for p in bad_phrases):
        return 0.2

    length_target = {"simple": 10, "medium": 20, "complex": 40}.get((complexity or "medium").strip().lower(), 20)
    length_score = min(len(answer.split()) / max(1, length_target), 1.0)
    context_words = set(context.lower().split())
    answer_words = set(answer_lower.split())
    if len(answer_words) == 0:
        return 0.0
    overlap = len(answer_words & context_words) / len(answer_words)

    score = 0.5 * length_score + 0.5 * overlap
    return round(score, 2)

# -----------------------------
# TF-IDF SIMILARITY
# -----------------------------
def tfidf_similarity(answer: str, context: str) -> float:
    if not answer.strip() or not context.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([answer, context])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(score, 0.0, 1.0))
    except Exception as e:
        print(f"[ConfidenceChecker] TF-IDF similarity failed: {e}")
        return 0.0

# -----------------------------
# EMBEDDING SIMILARITY
# -----------------------------
def embedding_similarity(answer: str, context: str) -> float:
    if not answer.strip() or not context.strip():
        return 0.0
    model = _get_embedding_model()
    if model is None:
        return 0.0
    try:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", context)
            if len(s.strip()) > 15
        ]
        if not sentences:
            sentences = [context.strip()]

        all_texts = [answer.strip()] + sentences
        embs = model.encode(all_texts, convert_to_numpy=True)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        answer_emb = embs[0:1]
        context_embs = embs[1:]

        sims = cosine_similarity(answer_emb, context_embs).flatten()
        if sims.size == 0:
            return 0.0
        return float(np.clip(np.max(sims), 0.0, 1.0))
    except Exception as e:
        print(f"[ConfidenceChecker] Embedding similarity failed: {e}")
        return 0.0

# -----------------------------
# RETRY WITH STRONGER MODEL
# -----------------------------
def retry_with_stronger_model(
    query: str,
    context: str,
    stronger_model: str,
    temperature: float = 0.0
) -> str:
    llm = ChatOpenAI(model=stronger_model, temperature=temperature)
    messages = [
        SystemMessage(
            content="You answer questions using ONLY the provided context. If the context is insufficient, say so clearly."
        ),
        HumanMessage(
            content=f"Question: {query}\n\nContext:\n{context}\n\nAnswer concisely."
        ),
    ]
    response = llm.invoke(messages)
    content = getattr(response, "content", "")

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                text_parts.append(item.text)
        return "\n".join(part.strip() for part in text_parts if part).strip()
    return str(content).strip()

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def check_confidence(
    query: str,
    compressed_context: str,
    router_output: Dict[str, Any],
    analyzer_output: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Return final answer (possibly retried), confidence scores, and metadata for logging."""

    _get_embedding_model()

    answer = router_output.get("answer", "")
    model_used_original = router_output.get("model_used", "")
    complexity = router_output.get("complexity", "medium")
    analyzer_confidence = (analyzer_output or {}).get("confidence")
    analyzer_source = (analyzer_output or {}).get("source")

    # Step 1 — heuristic score
    heuristic = heuristic_score(answer, compressed_context, complexity)

    # Step 2 — TF-IDF score
    tfidf = tfidf_similarity(answer, compressed_context)

    # Step 3 — Embedding similarity score
    embed = embedding_similarity(answer, compressed_context)

    # Combine weighted average: 50% TF-IDF + 50% embedding
    semantic_score = 0.5 * tfidf + 0.5 * embed

    # Final confidence = 70% semantic + 30% heuristic
    confidence_score_original = round(0.7 * semantic_score + 0.3 * heuristic, 2)
    confidence_threshold = _select_confidence_threshold(model_used_original, router_output)

    retried = False
    retry_reason = None
    model_used_final = model_used_original
    confidence_score_final = confidence_score_original
    model_tiers = _build_model_tiers()

    # Step 4 — retry if low confidence
    if confidence_score_final < confidence_threshold and model_used_original in model_tiers:
        retried = True
        low_scores = []
        if heuristic < confidence_threshold:
            low_scores.append(f"heuristic={heuristic}")
        if tfidf < confidence_threshold:
            low_scores.append(f"tfidf={tfidf}")
        if embed < confidence_threshold:
            low_scores.append(f"embedding={embed}")
        retry_reason = f"confidence {confidence_score_original} < threshold {confidence_threshold}" + (
            f" (low: {', '.join(low_scores)})" if low_scores else ""
        )

        stronger_model = model_tiers[model_used_original]
        try:
            new_answer = retry_with_stronger_model(
                query=query,
                context=compressed_context,
                stronger_model=stronger_model,
            )

            # recompute scores
            heuristic_new = heuristic_score(new_answer, compressed_context, complexity)
            tfidf_new = tfidf_similarity(new_answer, compressed_context)
            embed_new = embedding_similarity(new_answer, compressed_context)
            semantic_new = 0.5 * tfidf_new + 0.5 * embed_new
            final_new = round(0.7 * semantic_new + 0.3 * heuristic_new, 2)

            if final_new > confidence_score_final:
                answer = new_answer
                confidence_score_final = final_new
                model_used_final = stronger_model

        except Exception as e:
            print(f"[ConfidenceChecker] Retry failed: {e}")

    semantic = round(0.5 * tfidf + 0.5 * embed, 4)
    return {
        "final_answer": answer,
        "model_used_original": model_used_original,
        "model_used_final": model_used_final,
        "confidence_score_original": confidence_score_original,
        "confidence_score_final": confidence_score_final,
        "retried": retried,
        "retry_reason": retry_reason,
        "analyzer_confidence": analyzer_confidence,
        "analyzer_source": analyzer_source,
        "answer_complexity": complexity,
        "confidence_threshold": confidence_threshold,
        "confidence_threshold_local": CONFIDENCE_THRESHOLD_LOCAL,
        "confidence_threshold_openai": CONFIDENCE_THRESHOLD_OPENAI,
        "confidence_checker_embedding_enabled": _EMBEDDING_MODEL is not None,
        "confidence_semantic": semantic,
        "score_breakdown": {
            "heuristic": heuristic,
            "tfidf": tfidf,
            "embedding": embed,
            "semantic": semantic,
        },
    }