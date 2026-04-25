# Adaptive Retriever Update (One-Pager)

## Objective
Improve retrieval quality and cost-efficiency by moving from fixed label-based chunk retrieval to score-aware, confidence-aware dynamic retrieval.

## What Changed

### 1) Score-first retrieval sizing
- Retriever now accepts `complexity_score` (0-1) as primary input.
- Label (`simple|medium|complex`) is still accepted for backward compatibility.
- Base chunk count formula:
  - `k_base = round(3 + complexity_score * 7)` (bounded to `3..10`)

### 2) Confidence-aware safety bump
- If query analyzer confidence is low (`< 0.70`), retriever increases initial retrieval:
  - `k_final = min(k_base + 2, 12)`
- Purpose: reduce risk of under-retrieval for uncertain query complexity.

### 3) Retrieval quality check + one retry
- After retrieval, retriever computes keyword coverage:
  - `% of query keywords found in retrieved chunk text`
- If coverage is below threshold, retriever performs one re-retrieval step.
- Current tuned retry step: `+1` chunk (capped at `12`).

### 4) Complexity-aware retry thresholds
- Coverage thresholds are now class-specific:
  - `simple`: `0.45`
  - `medium`: `0.55`
  - `complex`: `0.60`
- Purpose: avoid over-retrying easy queries while protecting complex ones.

### 5) Confidence-gated retry
- Retry is only allowed when analyzer confidence is below `0.75` (or missing).
- Purpose: prevent unnecessary retries when analyzer is already confident.

## New Metadata Logged
Retriever now returns:
- `complexity_score_used`
- `analyzer_confidence`
- `k_base`, `k_final`
- `coverage_score`
- `coverage_threshold_used`
- `retry_confidence_gate_used`
- `retrieval_retry` (True/False)

These are propagated into pipeline/evaluation outputs for analysis.

## File Changes
- `src/adaptive_retriever.py`
  - Added score-first `k`, quality check, retry policy, richer metadata.
- `src/pipeline.py`
  - Passes analyzer score/confidence into retriever and returns new retrieval fields.
- `evaluate.py`
  - Logs new retriever metrics into CSV/JSON and prints summary stats.

## Current Results (50-query retrieval-only check)
- Before tuning:
  - `retrieval_retries = 20/50` (40%)
  - `avg_k_final = 5.24`
  - `avg_coverage ≈ 0.636`
- After tuning:
  - `retrieval_retries = 10/50` (20%)
  - `avg_k_final = 4.64`
  - `avg_coverage ≈ 0.634`

Interpretation:
- Retry rate and retrieval size dropped significantly.
- Coverage stayed nearly unchanged.
- This indicates better efficiency without meaningful quality loss in retrieval coverage.

## Notes / Known Constraint
- Full end-to-end `evaluate.py` currently requires a valid OpenAI key because answer generation uses `ChatOpenAI` in `src/model_router.py` (not for embeddings/retrieval).
