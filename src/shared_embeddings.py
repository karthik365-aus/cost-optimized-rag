"""
Shared sentence-transformer cache.

Keeps one in-process ``SentenceTransformer`` instance per model name so multiple
pipeline components can reuse the same loaded weights.
"""
import logging
from typing import Dict, Optional

LOG = logging.getLogger(__name__)

_MODELS: Dict[str, object] = {}
_FAILED_MODELS: set[str] = set()


def get_embedding_model(model_name: str):
    """
    Return a cached ``SentenceTransformer`` for ``model_name`` or ``None`` if loading fails.
    """
    if model_name in _FAILED_MODELS:
        LOG.debug("[Embeddings] skipping previously failed model load: %s", model_name)
        return None
    if model_name in _MODELS:
        LOG.debug("[Embeddings] reusing cached model: %s", model_name)
        return _MODELS[model_name]

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        _MODELS[model_name] = model
        LOG.debug("[Embeddings] loaded model: %s", model_name)
        return model
    except Exception as e:
        _FAILED_MODELS.add(model_name)
        LOG.debug("[Embeddings] failed to load model %s: %s", model_name, e)
        return None


def clear_embedding_cache(model_name: Optional[str] = None) -> None:
    """
    Clear all cached models, or one model when ``model_name`` is provided.
    """
    if model_name is None:
        _MODELS.clear()
        _FAILED_MODELS.clear()
        return
    _MODELS.pop(model_name, None)
    _FAILED_MODELS.discard(model_name)
