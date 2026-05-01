"""
Benchmark driver: run ``RAGPipeline`` on each row of ``data/test_queries.csv`` (or a
passed path), then write wide CSV + JSON for analysis.

**Ground truth here** (substring, token F1, embedding similarity vs ``ground_truth``)
measures whether the **final answer** matches the labeled benchmark span. That is
separate from ``src/confidence_checker``, which scores **answer vs compressed context**
for grounding and optional model retry.

Outputs default to project-root ``evaluation_results.{csv,json}``; pass paths to
``run_evaluation`` (or CLI flags) to version runs (e.g. under ``results/``). Optional
soft metrics use ``sentence_transformers`` when available; model name: env
``EVAL_GT_EMBEDDING_MODEL`` (default ``BAAI/bge-small-en-v1.5``).

CLI: ``python evaluate.py --help`` — ``--queries``, ``--out-csv``, ``--out-json``,
``--documents``, ``--seed``, ``--log-file``, ``--log-json``, ``--no-preflight``,
``--skip-local-model-check``, ``--require-openai-key``, ``-v`` / ``-q``.
"""
import argparse
import csv
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
# Preflight runs before RAGPipeline loads other modules — load `.env` here so
# LOCAL_* / OPENAI_* are visible to os.getenv during checks.
load_dotenv(PROJECT_ROOT / ".env")

from src.preflight import (
    PreflightError,
    csv_has_complex_queries,
    resolve_under_project,
    run_eval_preflight,
    validate_queries_csv,
)
from src.seed_manager import set_all_seeds
from src.shared_embeddings import get_embedding_model

LOG = logging.getLogger(__name__)


def configure_logging(log_level: int, log_file: str | None = None) -> None:
    """Console logging always; optional UTF-8 file under project or absolute path."""
    root = logging.getLogger()
    root.setLevel(log_level)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file:
        fp = resolve_under_project(PROJECT_ROOT, log_file)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(fp, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run RAGPipeline on each benchmark row; write CSV, JSON, and a printed report.",
    )
    p.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Benchmark CSV (default: data/test_queries.csv). Relative paths resolve from project root.",
    )
    p.add_argument("--out-csv", type=str, default=None, help="Output CSV path (default: evaluation_results.csv).")
    p.add_argument("--out-json", type=str, default=None, help="Output JSON path (default: evaluation_results.json).")
    p.add_argument(
        "--documents",
        type=str,
        default=None,
        help="Document folder for Chroma ingest when chroma_db is missing (default: data/documents).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--no-preflight", action="store_true", help="Skip vector DB, local /v1/models, and API key checks.")
    p.add_argument(
        "--skip-local-model-check",
        action="store_true",
        help="Do not call local server /v1/models (use if server is not up yet).",
    )
    p.add_argument(
        "--require-openai-key",
        action="store_true",
        help="Fail fast if OPENAI_API_KEY is unset (needed for complex-tier OpenAI calls).",
    )
    p.add_argument("--log-file", type=str, default=None, help="Append structured logs to this file.")
    p.add_argument(
        "--log-json",
        type=str,
        default=None,
        help="Append one JSON object per line (query_complete events) to this path.",
    )
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v for INFO, -vv for DEBUG on loggers.")
    p.add_argument("-q", "--quiet", action="store_true", help="Warnings and errors only (logging.WARNING).")
    return p


# -----------------------------
# GROUND-TRUTH APPROPRIATENESS (beyond strict substring)
# -----------------------------
_EVAL_EMBED_MODEL = None
_EVAL_EMBED_FAILED = False


def _ground_truth_evaluable(ground_truth: str) -> bool:
    if not (ground_truth or "").strip():
        return False
    g = ground_truth.lower().strip()
    return g not in ("manual evaluation needed", "requires multi-step analysis")


def _word_counter(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", (text or "").lower()))


def answer_ground_truth_token_f1(ground_truth: str, llm_answer: str):
    """Multiset token F1 between ground truth span and full answer (0..1). None if not evaluable."""
    if not _ground_truth_evaluable(ground_truth):
        return None
    cg = _word_counter(ground_truth)
    ca = _word_counter(llm_answer)
    if not cg or not ca:
        return None
    overlap = sum(min(cg[w], ca[w]) for w in cg.keys())
    if overlap <= 0:
        return 0.0
    prec = overlap / max(sum(ca.values()), 1)
    rec = overlap / max(sum(cg.values()), 1)
    if prec + rec <= 0:
        return 0.0
    return round(2 * prec * rec / (prec + rec), 4)


def _eval_embedding_model():
    global _EVAL_EMBED_MODEL, _EVAL_EMBED_FAILED
    if _EVAL_EMBED_FAILED:
        return None
    if _EVAL_EMBED_MODEL is not None:
        return _EVAL_EMBED_MODEL
    try:
        name = os.getenv("EVAL_GT_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        _EVAL_EMBED_MODEL = get_embedding_model(name)
        if _EVAL_EMBED_MODEL is None:
            _EVAL_EMBED_FAILED = True
            return None
    except Exception:
        _EVAL_EMBED_FAILED = True
        return None
    return _EVAL_EMBED_MODEL


def answer_ground_truth_embedding_similarity(ground_truth: str, llm_answer: str):
    """Cosine similarity of sentence embeddings (0..1). None if not evaluable or model unavailable."""
    if not _ground_truth_evaluable(ground_truth):
        return None
    if not (llm_answer or "").strip():
        return None
    model = _eval_embedding_model()
    if model is None:
        return None
    try:
        emb = model.encode(
            [ground_truth.strip(), llm_answer.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        sim = float(cosine_similarity(emb[0:1], emb[1:2])[0][0])
        return round(max(0.0, min(1.0, sim)), 4)
    except TypeError:
        emb = model.encode([ground_truth.strip(), llm_answer.strip()], convert_to_numpy=True)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        sim = float(cosine_similarity(emb[0:1], emb[1:2])[0][0])
        return round(max(0.0, min(1.0, sim)), 4)
    except Exception:
        return None


def answer_appropriateness_score(f1_val, sim_val):
    """Single 0..1 blend for reporting. Uses whichever signals exist."""
    if f1_val is None and sim_val is None:
        return None
    if f1_val is None:
        return round(float(sim_val), 4)
    if sim_val is None:
        return round(float(f1_val), 4)
    return round(0.5 * float(f1_val) + 0.5 * float(sim_val), 4)


# -----------------------------
# ANSWER CORRECTNESS
# -----------------------------
def is_answer_correct(ground_truth: str, llm_answer: str) -> bool:
    """Case-insensitive substring check — ground truth must appear in the LLM answer."""
    if not ground_truth or ground_truth.lower() in ("manual evaluation needed", "requires multi-step analysis"):
        return None  # can't auto-evaluate
    return ground_truth.lower() in llm_answer.lower()


# -----------------------------
# FIND WHICH SENTENCE CONTAINS THE ANSWER
# -----------------------------
def find_answer_sentence_index(sentences: list, ground_truth: str) -> int:
    """Return index of the first sentence containing the ground truth answer, or -1."""
    if not ground_truth:
        return -1
    for i, sentence in enumerate(sentences):
        if ground_truth.lower() in sentence.lower():
            return i
    return -1


# -----------------------------
# FAILURE CATEGORIZATION
# -----------------------------
def categorize_failure(
    ground_truth_source: str,
    retrieved_sources: list,
    all_sentences: list,
    selected_indices: list,
    ground_truth: str,
    llm_answer: str,
) -> str:
    """When substring match fails: TYPE_A retrieval, TYPE_B compression dropped sentence, TYPE_C LLM."""
    correct = is_answer_correct(ground_truth, llm_answer)

    # Can't categorize if no ground truth source or manual evaluation needed
    if not ground_truth_source or correct is None:
        return "N/A"

    if correct:
        return "CORRECT"

    # Type A: correct chunk was never retrieved
    if not any(ground_truth_source in src for src in retrieved_sources):
        return "TYPE_A"

    # Type B: chunk retrieved but answer sentence was compressed away
    answer_sentence_idx = find_answer_sentence_index(all_sentences, ground_truth)
    if answer_sentence_idx != -1 and answer_sentence_idx not in selected_indices:
        return "TYPE_B"

    # Type C: context had the answer, LLM got it wrong
    return "TYPE_C"


# -----------------------------
# MAIN EVALUATION
# -----------------------------
def run_evaluation(
    queries_file: str = None,
    output_csv: str = None,
    output_json: str = None,
    documents_path: str = None,
    preflight: bool = True,
    skip_local_model_check: bool = False,
    require_openai_key: bool = False,
    json_log_path: str | None = None,
):
    """Load CSV queries, run pipeline per row, write CSV/JSON and print aggregate report.

    ``queries_file`` rows need at least ``query_id``, ``query``, ``complexity``,
    ``ground_truth``; ``ground_truth_source`` enables retrieval_hit and failure typing.

    Paths may be relative to the project root. When ``preflight`` is True, validates
    CSV, Chroma or ingestible documents, optional local ``/v1/models``, and optionally
    requires ``OPENAI_API_KEY``.
    """
    queries_file = queries_file or str(PROJECT_ROOT / "data" / "test_queries.csv")
    output_csv = output_csv or str(PROJECT_ROOT / "evaluation_results.csv")
    output_json = output_json or str(PROJECT_ROOT / "evaluation_results.json")
    documents_path = documents_path or str(PROJECT_ROOT / "data" / "documents")

    queries_path = resolve_under_project(PROJECT_ROOT, queries_file)
    out_csv_path = resolve_under_project(PROJECT_ROOT, output_csv)
    out_json_path = resolve_under_project(PROJECT_ROOT, output_json)
    docs_path = resolve_under_project(PROJECT_ROOT, documents_path)

    row_count = validate_queries_csv(queries_path)
    has_complex_queries = csv_has_complex_queries(queries_path)
    LOG.info("Benchmark CSV OK: %s data rows in %s", row_count, queries_path)

    if preflight:
        run_eval_preflight(
            project_root=PROJECT_ROOT,
            documents_path=docs_path,
            skip_local_model_check=skip_local_model_check,
            require_openai_key=require_openai_key,
            has_complex_queries=has_complex_queries,
        )
    else:
        LOG.warning(
            "Preflight disabled: not checking Chroma/documents, local /v1/models, or OPENAI_API_KEY."
        )

    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline(documents_path=str(docs_path))

    rows = []
    json_records = []
    json_log_fp = None

    # --- Per query: full pipeline + benchmark metrics ---
    try:
        if json_log_path:
            jpath = resolve_under_project(PROJECT_ROOT, json_log_path)
            jpath.parent.mkdir(parents=True, exist_ok=True)
            json_log_fp = open(jpath, "a", encoding="utf-8")

        with open(queries_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                query_id = row["query_id"]
                query = row["query"]
                complexity = row["complexity"]
                ground_truth = row["ground_truth"].strip()
                ground_truth_source = row.get("ground_truth_source", "").strip()
    
                result = pipeline.run(query)
    
                retrieved_sources = [c["source"] for c in result["chunks"]]
                all_sentences = result["all_sentences"]
                selected_indices = result["selected_indices"]
                llm_answer = result["final_answer"]
    
                correct = is_answer_correct(ground_truth, llm_answer)
                gt_f1 = answer_ground_truth_token_f1(ground_truth, llm_answer)
                gt_sim = answer_ground_truth_embedding_similarity(ground_truth, llm_answer)
                gt_appropriate = answer_appropriateness_score(gt_f1, gt_sim)
                answer_sentence_idx = find_answer_sentence_index(all_sentences, ground_truth)
                failure_type = categorize_failure(
                    ground_truth_source,
                    retrieved_sources,
                    all_sentences,
                    selected_indices,
                    ground_truth,
                    llm_answer,
                )
                retrieval_hit = (
                    "yes" if any(ground_truth_source in src for src in retrieved_sources)
                    else "no" if ground_truth_source else "n/a"
                )
    
                _comp_meta = result.get("compression_metadata") or {}
                _hybrid = _comp_meta.get("hybrid_weights") or {}
                _reason_codes = result.get("complexity_reason_codes") or []
                _reason_str = "|".join(str(x) for x in _reason_codes) if isinstance(_reason_codes, list) else str(_reason_codes)
                _routed = result.get("complexity", "")
                _rerouted = result.get("model_rerouted", False)
    
                csv_row = {
                    "query_id": query_id,
                    "query": query,
                    "complexity": complexity,
                    "complexity_routed": _routed,
                    "complexity_label_match": ("yes" if complexity == _routed else "no") if complexity else "",
                    "complexity_score": result.get("complexity_score", ""),
                    "complexity_source": result.get("complexity_source", ""),
                    "complexity_confidence": result.get("complexity_confidence", ""),
                    "complexity_reason_codes": _reason_str,
                    "query_word_count": result.get("query_word_count", ""),
                    "query_analyzer_llm_status": result.get("query_analyzer_llm_status", ""),
                    "query_analyzer_ms": result.get("query_analyzer_ms", ""),
                    "retrieval_ms": result.get("retrieval_ms", ""),
                    "compression_ms": result.get("compression_ms", ""),
                    "model_router_ms": result.get("model_router_ms", ""),
                    "confidence_checker_ms": result.get("confidence_checker_ms", ""),
                    "total_pipeline_ms": result.get("total_pipeline_ms", ""),
                    "ground_truth": ground_truth,
                    "ground_truth_source": ground_truth_source,
                    "answer_correct": "" if correct is None else ("yes" if correct else "no"),
                    "answer_gt_token_f1": "" if gt_f1 is None else gt_f1,
                    "answer_gt_embedding_similarity": "" if gt_sim is None else gt_sim,
                    "answer_gt_appropriateness_score": "" if gt_appropriate is None else gt_appropriate,
                    "failure_type": failure_type,
                    "retrieved_sources": " | ".join(retrieved_sources),
                    "retrieval_hit": retrieval_hit,
                    "answer_sentence_idx": answer_sentence_idx,
                    "answer_sentence_selected": (
                        "yes" if answer_sentence_idx in selected_indices
                        else "no" if answer_sentence_idx != -1
                        else "n/a"
                    ),
                    "k": result["k"],
                    "k_base": result.get("k_base", ""),
                    "k_final": result.get("k_final", ""),
                    "retrieval_complexity_score_used": result.get("retrieval_complexity_score_used", ""),
                    "retrieval_analyzer_confidence": result.get("retrieval_analyzer_confidence", ""),
                    "coverage_score": result.get("coverage_score", ""),
                    "coverage_threshold_used": result.get("coverage_threshold_used", ""),
                    "retrieval_retry_confidence_gate": result.get("retrieval_retry_confidence_gate", ""),
                    "retrieval_avg_chunk_distance": result.get("retrieval_avg_chunk_distance", ""),
                    "retrieval_retry": result.get("retrieval_retry", False),
                    "factual_fastpath_attempted": result.get("factual_fastpath_attempted", False),
                    "factual_fastpath_used": result.get("factual_fastpath_used", False),
                    "rerank_enabled": result.get("rerank_enabled", False),
                    "rerank_used": result.get("rerank_used", False),
                    "rerank_model": result.get("rerank_model", ""),
                    "rerank_latency_ms": result.get("rerank_latency_ms", ""),
                    "original_tokens": result["original_token_count"],
                    "compressed_tokens": result["compressed_token_count"],
                    "compression_ratio": result["compression_ratio"],
                "compression_adaptive_top_n": _comp_meta.get("adaptive_top_n", ""),
                "compression_sentence_budget_base": _comp_meta.get("sentence_budget_base", ""),
                "compression_coverage_score_used": _comp_meta.get("coverage_score_used", ""),
                "compression_coverage_budget_extra": _comp_meta.get("coverage_budget_extra", ""),
                "compression_used_embeddings": _comp_meta.get("used_embeddings", ""),
                "compression_score_used": _comp_meta.get("complexity_score_used", ""),
                    "compression_hybrid_tfidf_w": _hybrid.get("tfidf", ""),
                    "compression_hybrid_embedding_w": _hybrid.get("embedding", ""),
                    "compression_redundancy_skips": _comp_meta.get("redundancy_skips", ""),
                    "compression_sentence_count": result.get("compression_sentence_count", ""),
                    "compression_selected_sentence_count": result.get("compression_selected_sentence_count", ""),
                    "compression_tokens_saved": result.get("compression_tokens_saved", ""),
                    "pipeline_tokens_context_to_llm_est": result.get("pipeline_tokens_context_to_llm_est", ""),
                    "pipeline_tokens_llm_total_est": result.get("pipeline_tokens_llm_total_est", ""),
                    "router_model_source": result.get("router_model_source", ""),
                    "router_fallback_reason": result.get("router_fallback_reason", ""),
                    "router_total_tokens_est": result.get("router_total_tokens_est", ""),
                    "model_original": result["model_used_original"],
                    "model_final": result["model_used_final"],
                    "model_rerouted": "yes" if _rerouted else "no",
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "time_seconds": result["time_taken_seconds"],
                    "confidence_original": result["confidence_score_original"],
                    "confidence_final": result["confidence_score_final"],
                    "confidence_semantic": result.get("confidence_semantic", ""),
                    "confidence_threshold": result.get("confidence_threshold", ""),
                    "confidence_checker_embedding_enabled": result.get("confidence_checker_embedding_enabled", ""),
                    "retried": result["retried"],
                    "retry_reason": result.get("retry_reason") or "",
                    "heuristic_score": result["score_breakdown"]["heuristic"],
                    "tfidf_score": result["score_breakdown"]["tfidf"],
                    "embedding_score": result["score_breakdown"]["embedding"],
                    "final_answer": llm_answer,
                }
                rows.append(csv_row)
    
                # Full JSON record with all sentences and scores for deep-dive analysis
                json_records.append({
                    **csv_row,
                    "all_sentences": all_sentences,
                    "sentence_scores": result["sentence_scores"],
                    "selected_indices": selected_indices,
                    "chunks": result["chunks"],
                    "compression_metadata": result.get("compression_metadata") or {},
                    "pipeline_stage_summary": {
                        "analyzer_ms": result.get("query_analyzer_ms"),
                        "retrieval_ms": result.get("retrieval_ms"),
                        "compression_ms": result.get("compression_ms"),
                        "router_ms": result.get("model_router_ms"),
                        "confidence_ms": result.get("confidence_checker_ms"),
                        "total_ms": result.get("total_pipeline_ms"),
                        "tokens": {
                            "retrieved_context_est": result.get("original_token_count"),
                            "compressed_context_est": result.get("compressed_token_count"),
                            "saved_by_compression": result.get("compression_tokens_saved"),
                            "router_prompt_plus_completion_est": result.get("router_total_tokens_est"),
                        },
                        "routing": {
                            "complexity_routed": _routed,
                            "label_match_vs_test_row": csv_row["complexity_label_match"],
                            "router_model_source": result.get("router_model_source"),
                            "model_rerouted": _rerouted,
                        },
                        "confidence": {
                            "semantic": result.get("confidence_semantic"),
                            "final": result.get("confidence_score_final"),
                            "threshold": result.get("confidence_threshold"),
                            "checker_embedding_enabled": result.get("confidence_checker_embedding_enabled"),
                            "retried": result.get("retried"),
                        },
                        "retrieval": {
                            "retry": result.get("retrieval_retry"),
                            "coverage": result.get("coverage_score"),
                            "coverage_threshold": result.get("coverage_threshold_used"),
                        },
                        "answer_vs_ground_truth": {
                            "substring_hit": csv_row["answer_correct"],
                            "token_f1": gt_f1,
                            "embedding_similarity": gt_sim,
                            "appropriateness_score": gt_appropriate,
                        },
                    },
                })

                if json_log_fp is not None:
                    json_log_fp.write(
                        json.dumps(
                            {
                                "event": "query_complete",
                                "query_id": query_id,
                                "total_pipeline_ms": result.get("total_pipeline_ms"),
                                "substring_correct": csv_row["answer_correct"],
                                "failure_type": failure_type,
                            },
                            default=str,
                        )
                        + "\n"
                    )
                LOG.info(
                    "query_complete query_id=%s total_ms=%s substring=%s",
                    query_id,
                    result.get("total_pipeline_ms"),
                    csv_row["answer_correct"],
                )

    finally:
        if json_log_fp is not None:
            json_log_fp.close()

    if not rows:
        raise RuntimeError("No query rows were processed; check the CSV.")

    # --- Persist ---
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    # Write CSV
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Write JSON
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2)

    # --- Summary ---
    evaluatable = [r for r in rows if r["answer_correct"] in ("yes", "no")]
    correct_count = sum(1 for r in evaluatable if r["answer_correct"] == "yes")
    wrong_count = sum(1 for r in evaluatable if r["answer_correct"] == "no")

    type_a = sum(1 for r in rows if r["failure_type"] == "TYPE_A")
    type_b = sum(1 for r in rows if r["failure_type"] == "TYPE_B")
    type_c = sum(1 for r in rows if r["failure_type"] == "TYPE_C")

    retrieval_evaluatable = [r for r in rows if r["retrieval_hit"] in ("yes", "no")]
    retrieval_correct = sum(1 for r in retrieval_evaluatable if r["retrieval_hit"] == "yes")

    retried_count = sum(1 for r in rows if str(r["retried"]).lower() == "true")
    flan_invocations = sum(1 for r in rows if r["query_analyzer_llm_status"] != "not_called")
    flan_timeouts = sum(1 for r in rows if r["query_analyzer_llm_status"] == "timeout")
    flan_errors = sum(1 for r in rows if r["query_analyzer_llm_status"] in ("runtime_error", "parse_error"))
    retrieval_retries = sum(1 for r in rows if str(r.get("retrieval_retry")).lower() == "true")
    factual_fastpath_attempted = sum(1 for r in rows if str(r.get("factual_fastpath_attempted")).lower() == "true")
    factual_fastpath_used = sum(1 for r in rows if str(r.get("factual_fastpath_used")).lower() == "true")
    rerank_used = sum(1 for r in rows if str(r.get("rerank_used")).lower() == "true")
    avg_coverage = round(
        sum(float(r["coverage_score"]) for r in rows if str(r["coverage_score"]) != "")
        / max(1, sum(1 for r in rows if str(r["coverage_score"]) != "")),
        4,
    )
    avg_compression = round(sum(float(r["compression_ratio"]) for r in rows) / len(rows), 4)
    avg_confidence = round(sum(float(r["confidence_final"]) for r in rows) / len(rows), 4)
    avg_query_analyzer_ms = round(sum(float(r["query_analyzer_ms"]) for r in rows if str(r["query_analyzer_ms"]) != "") / len(rows), 2)
    avg_retrieval_ms = round(sum(float(r["retrieval_ms"]) for r in rows if str(r["retrieval_ms"]) != "") / len(rows), 2)
    avg_compression_ms = round(sum(float(r["compression_ms"]) for r in rows if str(r["compression_ms"]) != "") / len(rows), 2)
    avg_model_router_ms = round(sum(float(r["model_router_ms"]) for r in rows if str(r["model_router_ms"]) != "") / len(rows), 2)
    avg_confidence_checker_ms = round(sum(float(r["confidence_checker_ms"]) for r in rows if str(r["confidence_checker_ms"]) != "") / len(rows), 2)
    avg_total_pipeline_ms = round(sum(float(r["total_pipeline_ms"]) for r in rows if str(r["total_pipeline_ms"]) != "") / len(rows), 2)
    avg_tokens_saved = round(
        sum(float(r["compression_tokens_saved"]) for r in rows if str(r.get("compression_tokens_saved", "")) != "")
        / max(1, sum(1 for r in rows if str(r.get("compression_tokens_saved", "")) != "")),
        2,
    )
    avg_confidence_semantic = round(
        sum(float(r["confidence_semantic"]) for r in rows if str(r.get("confidence_semantic", "")) != "")
        / max(1, sum(1 for r in rows if str(r.get("confidence_semantic", "")) != "")),
        4,
    )
    model_reroute_count = sum(1 for r in rows if str(r.get("model_rerouted", "")).lower() == "yes")

    gt_quality_rows = [r for r in rows if str(r.get("answer_gt_appropriateness_score", "")) != ""]
    gt_f1_rows = [r for r in rows if str(r.get("answer_gt_token_f1", "")) != ""]
    avg_gt_f1 = (
        round(sum(float(r["answer_gt_token_f1"]) for r in gt_f1_rows) / len(gt_f1_rows), 4) if gt_f1_rows else None
    )
    gt_sim_rows = [r for r in rows if str(r.get("answer_gt_embedding_similarity", "")) != ""]
    avg_gt_sim = (
        round(sum(float(r["answer_gt_embedding_similarity"]) for r in gt_sim_rows) / len(gt_sim_rows), 4)
        if gt_sim_rows
        else None
    )
    avg_gt_appropriate = (
        round(sum(float(r["answer_gt_appropriateness_score"]) for r in gt_quality_rows) / len(gt_quality_rows), 4)
        if gt_quality_rows
        else None
    )

    avg_orig_tokens = round(sum(float(r["original_tokens"]) for r in rows) / len(rows), 2)
    avg_comp_tokens = round(sum(float(r["compressed_tokens"]) for r in rows) / len(rows), 2)
    avg_router_tokens = round(
        sum(float(r["router_total_tokens_est"]) for r in rows if str(r.get("router_total_tokens_est", "")) != "")
        / max(1, sum(1 for r in rows if str(r.get("router_total_tokens_est", "")) != "")),
        2,
    )

    print(f"\n{'='*60}")
    print(f"FAILURE ANALYSIS REPORT")
    print(f"{'='*60}")
    print(f"Total queries run       : {len(rows)}")
    print(f"Auto-evaluatable        : {len(evaluatable)} (rows with ground truth usable for strict substring check)")
    if evaluatable:
        accuracy = round(correct_count / len(evaluatable) * 100, 1)
        print(f"Substring hit rate      : {correct_count}/{len(evaluatable)} ({accuracy}%)  [ground_truth must appear in answer]")
        print(f"Substring miss          : {wrong_count}/{len(evaluatable)}")
    if retrieval_evaluatable:
        ret_acc = round(retrieval_correct / len(retrieval_evaluatable) * 100, 1)
        print(f"Retrieval accuracy      : {retrieval_correct}/{len(retrieval_evaluatable)} ({ret_acc}%)")

    if wrong_count > 0:
        print(f"\nBREAKDOWN OF {wrong_count} FAILURES:")
        print(f"  Type A (Retrieval)    : {type_a} ({round(type_a/wrong_count*100,1)}%) — wrong chunks retrieved")
        print(f"  Type B (Compression)  : {type_b} ({round(type_b/wrong_count*100,1)}%) — answer sentence dropped")
        print(f"  Type C (Model/LLM)    : {type_c} ({round(type_c/wrong_count*100,1)}%) — context ok, LLM wrong")

        insights = []
        if type_a >= type_b and type_a >= type_c:
            insights.append("Biggest problem: Retrieval — Hybrid RAG (BM25+vector) would help Type A")
        if type_b >= type_a and type_b >= type_c:
            insights.append("Biggest problem: Compression — increase sentence retention to help Type B")
        if type_c >= type_a and type_c >= type_b:
            insights.append("Biggest problem: LLM — prompt engineering or stronger model would help Type C")

        print(f"\nINSIGHTS:")
        for insight in insights:
            print(f"  - {insight}")

    print(f"\nANSWER VS GROUND TRUTH (softer metrics)")
    print(f"  Rows with appropriateness score : {len(gt_quality_rows)}/{len(rows)}")
    if gt_quality_rows:
        print(f"  Avg token F1 (answer vs GT)      : {avg_gt_f1 if avg_gt_f1 is not None else 'n/a'}")
        if avg_gt_sim is not None:
            print(f"  Avg embedding sim (answer vs GT) : {avg_gt_sim}  (requires sentence-transformers)")
        else:
            print(f"  Avg embedding sim (answer vs GT) : n/a (model unavailable or all failed)")
        print(f"  Avg appropriateness score        : {avg_gt_appropriate}  (0.5*F1 + 0.5*embedding when both exist)")

    print(f"\nOTHER METRICS:")
    print(f"  Queries retried         : {retried_count}/{len(rows)}")
    print(f"  FLAN invocations        : {flan_invocations}/{len(rows)}")
    print(f"  FLAN timeouts           : {flan_timeouts}")
    print(f"  FLAN errors             : {flan_errors}")
    print(f"  Retrieval retries       : {retrieval_retries}/{len(rows)}")
    print(f"  Factual fastpath        : used {factual_fastpath_used}/{len(rows)} (attempted {factual_fastpath_attempted})")
    print(f"  Cross-encoder rerank    : used {rerank_used}/{len(rows)}")
    print(f"  Avg coverage score      : {avg_coverage}")
    print(f"  Avg compression ratio   : {avg_compression}")
    print(f"  Avg confidence score    : {avg_confidence}")
    print(f"  Avg query analyzer ms   : {avg_query_analyzer_ms}")
    print(f"  Avg retrieval ms        : {avg_retrieval_ms}")
    print(f"  Avg compression ms      : {avg_compression_ms}")
    print(f"  Avg model router ms     : {avg_model_router_ms}")
    print(f"  Avg confidence check ms : {avg_confidence_checker_ms}")
    print(f"  Avg total pipeline ms   : {avg_total_pipeline_ms}")
    print(f"  Avg tokens saved (compression): {avg_tokens_saved}")
    print(f"  Avg orig / compressed tokens : {avg_orig_tokens} / {avg_comp_tokens}")
    print(f"  Avg router tokens (est.)     : {avg_router_tokens}")
    print(f"  Avg confidence semantic : {avg_confidence_semantic}")
    print(f"  Model reroutes (confidence tier): {model_reroute_count}/{len(rows)}")
    print(f"\nResults saved to:")
    print(f"  CSV : {out_csv_path}")
    print(f"  JSON: {out_json_path}")
    LOG.info("Wrote CSV=%s JSON=%s rows=%s", out_csv_path, out_json_path, len(rows))


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    configure_logging(log_level, args.log_file)
    set_all_seeds(args.seed)
    try:
        run_evaluation(
            queries_file=args.queries,
            output_csv=args.out_csv,
            output_json=args.out_json,
            documents_path=args.documents,
            preflight=not args.no_preflight,
            skip_local_model_check=args.skip_local_model_check,
            require_openai_key=args.require_openai_key,
            json_log_path=args.log_json,
        )
    except PreflightError as exc:
        LOG.error("%s", exc)
        sys.exit(2)
    except KeyboardInterrupt:
        LOG.error("Interrupted.")
        sys.exit(130)
    except Exception as exc:
        LOG.exception("Evaluation failed: %s", exc)
        sys.exit(1)
