"""
Startup checks and a single place to log effective configuration (sanitized).

Used by ``evaluate.py`` before a long batch and by ``RAGPipeline`` after wiring
components so local/OpenAI settings match expectations.
"""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, Set

import requests

logger = logging.getLogger(__name__)

REQUIRED_QUERY_COLUMNS = frozenset({"query_id", "query", "complexity", "ground_truth"})


class PreflightError(Exception):
    """Raised when mandatory checks fail before running the pipeline."""


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_under_project(project_root: Path, path: Path | str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p


def chroma_dir(project_root: Path) -> Path:
    return project_root / "chroma_db"


def validate_queries_csv(queries_file: Path) -> int:
    """Ensure the benchmark CSV exists, has required headers, and at least one data row."""
    if not queries_file.is_file():
        raise PreflightError(f"Queries file not found: {queries_file}")
    with open(queries_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise PreflightError("CSV has no header row.")
        fields = {h.strip() for h in reader.fieldnames if h}
        missing = REQUIRED_QUERY_COLUMNS - fields
        if missing:
            raise PreflightError(f"CSV missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise PreflightError("Queries CSV has a header but zero data rows.")
    return len(rows)


def csv_has_complex_queries(queries_file: Path) -> bool:
    """Return True when any row has complexity label ``complex`` (case-insensitive)."""
    with open(queries_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            complexity = str((row or {}).get("complexity", "")).strip().lower()
            if complexity == "complex":
                return True
    return False


def check_vector_store(project_root: Path, documents_path: Path) -> None:
    """Require either an existing Chroma dir or ingestible ``.txt`` documents."""
    cdir = chroma_dir(project_root)
    if cdir.exists():
        return
    if not documents_path.is_dir():
        raise PreflightError(
            f"No vector DB at {cdir} and documents directory missing: {documents_path}"
        )
    txts = list(documents_path.rglob("*.txt"))
    if not txts:
        raise PreflightError(
            f"No .txt files under {documents_path}; cannot build Chroma at {cdir}."
        )


def _local_models_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def fetch_local_model_ids(base_url: str, timeout: float = 3.0) -> Set[str]:
    url = _local_models_url(base_url)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data", [])
    return {
        str(item.get("id", "")).strip()
        for item in data
        if isinstance(item, dict) and item.get("id")
    }


def check_local_openai_models(
    *,
    skip: bool,
    base_url: str,
    simple_model: str,
    medium_model: str,
    timeout: float,
) -> None:
    if skip:
        logger.info("Skipping local /v1/models preflight (--skip-local-model-check).")
        return
    try:
        ids = fetch_local_model_ids(base_url, timeout=timeout)
    except Exception as exc:
        raise PreflightError(
            f"Could not reach local OpenAI-compatible server at {base_url!r}: {exc}. "
            "Start LM Studio (or your proxy), verify LOCAL_OPENAI_BASE_URL, or pass --skip-local-model-check."
        ) from exc
    missing = [m for m in (simple_model, medium_model) if m and m not in ids]
    if missing:
        avail = ", ".join(sorted(ids)) if ids else "(none reported)"
        raise PreflightError(
            f"Local server does not list configured model(s) {missing}. "
            f"Loaded IDs: {avail}. Match LOCAL_SIMPLE_MODEL / LOCAL_MEDIUM_MODEL to /v1/models."
        )


def warn_openai_key_if_missing(require: bool) -> None:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if key:
        return
    msg = "OPENAI_API_KEY is unset; complex queries will fail against OpenAI."
    if require:
        raise PreflightError(msg)
    logger.warning(msg)


def effective_config_dict(project_root: Path, documents_path: str) -> Dict[str, Any]:
    """Snapshot of non-secret settings for logs and run_config artifacts."""
    key = os.getenv("OPENAI_API_KEY", "")
    return {
        "project_root": str(project_root),
        "documents_path": documents_path,
        "chroma_dir": str(chroma_dir(project_root)),
        "local_openai_base_url": os.getenv("LOCAL_OPENAI_BASE_URL", "http://localhost:1234/v1"),
        "local_simple_model": os.getenv("LOCAL_SIMPLE_MODEL", "tinyllama"),
        "local_medium_model": os.getenv("LOCAL_MEDIUM_MODEL", "ministral"),
        "openai_complex_model": os.getenv("OPENAI_COMPLEX_MODEL", "gpt-4o-mini"),
        "local_healthcheck_enabled": os.getenv("LOCAL_HEALTHCHECK_ENABLED", "true"),
        "compression_use_embeddings": os.getenv("COMPRESSION_USE_EMBEDDINGS", "true"),
        "compression_embedding_model": os.getenv("COMPRESSION_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "eval_gt_embedding_model": os.getenv("EVAL_GT_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "openai_api_key_set": bool(key.strip()),
        "hf_home": os.getenv("HF_HOME", ""),
    }


def log_effective_configuration(
    project_root: Path,
    documents_path: str,
    *,
    emit_print: bool = True,
) -> Dict[str, Any]:
    cfg = effective_config_dict(project_root, documents_path)
    lines = [
        "Effective configuration (sanitized):",
        f"  project_root: {cfg['project_root']}",
        f"  documents_path: {cfg['documents_path']}",
        f"  chroma_dir: {cfg['chroma_dir']}",
        f"  local_openai_base_url: {cfg['local_openai_base_url']}",
        f"  LOCAL_SIMPLE_MODEL: {cfg['local_simple_model']}",
        f"  LOCAL_MEDIUM_MODEL: {cfg['local_medium_model']}",
        f"  OPENAI_COMPLEX_MODEL: {cfg['openai_complex_model']}",
        f"  OPENAI_API_KEY set: {cfg['openai_api_key_set']}",
        f"  LOCAL_HEALTHCHECK_ENABLED: {cfg['local_healthcheck_enabled']}",
        f"  COMPRESSION_USE_EMBEDDINGS: {cfg['compression_use_embeddings']}",
        f"  COMPRESSION_EMBEDDING_MODEL: {cfg['compression_embedding_model']}",
        f"  EVAL_GT_EMBEDDING_MODEL: {cfg['eval_gt_embedding_model']}",
    ]
    if cfg.get("hf_home"):
        lines.append(f"  HF_HOME: {cfg['hf_home']}")
    text = "\n".join(lines)
    logger.info(text)
    if emit_print:
        print(text)
    return cfg


def run_eval_preflight(
    *,
    project_root: Path,
    documents_path: Path,
    skip_local_model_check: bool,
    require_openai_key: bool,
    has_complex_queries: bool = False,
) -> None:
    """Environment and data-path checks (call after ``validate_queries_csv``)."""
    check_vector_store(project_root, documents_path)
    warn_openai_key_if_missing(require_openai_key or has_complex_queries)
    base = os.getenv("LOCAL_OPENAI_BASE_URL", "http://localhost:1234/v1")
    timeout = float(os.getenv("LOCAL_HEALTHCHECK_TIMEOUT_SECONDS", "3"))
    simple = os.getenv("LOCAL_SIMPLE_MODEL", "tinyllama")
    medium = os.getenv("LOCAL_MEDIUM_MODEL", "ministral")
    check_local_openai_models(
        skip=skip_local_model_check,
        base_url=base,
        simple_model=simple,
        medium_model=medium,
        timeout=timeout,
    )


def log_pipeline_startup(project_root: Path, documents_path: str) -> None:
    """Log once after ``RAGPipeline`` constructs subsystems."""
    log_effective_configuration(project_root, documents_path, emit_print=True)
