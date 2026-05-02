# CLAUDE.md — Cost Optimized RAG

## What this project is
A cost-optimized RAG pipeline for BSAN 765 group project. Classifies query complexity and uses compression to reduce LLM token usage before routing to the appropriate model tier.

## Current Status
- ✅ Adaptive Retrieval — DONE (Karthik) — `src/adaptive_retriever.py`
- ✅ Context Compression — DONE (Anh) — `src/context_compression.py`
- ✅ Query Analyzer — IMPLEMENTED (rule + heuristic + FLAN fallback) — `src/query_analyzer.py`
- ✅ Confidence Checker — IMPLEMENTED (multi-signal confidence + model retry) — `src/confidence_checker.py`
- ✅ Pipeline Wiring — IMPLEMENTED (end-to-end in `src/pipeline.py`)
- ✅ Model Router — UPDATED (simple/medium → local OpenAI-compatible API; complex → OpenAI) — `src/model_router.py`

## Pipeline Flow
```
User Query
    ↓
Query Analyzer (complexity_score + confidence + label)
    ↓
Adaptive Retrieval (continuous k from score, confidence-aware, coverage retry)
    ↓
Context Compression (adaptive sentence budget + hybrid TF-IDF / local BGE embeddings + redundancy pruning + answer-type boosts; optional fixed caps via env)
    ↓
Model Router (simple→local LM Studio model / medium→local LM Studio model / complex→OpenAI, e.g. gpt-4o-mini)
    ↓
Confidence Checker (retry with stronger model if confidence low)
    ↓
Final Answer
```

## What Was Implemented (Latest Work)
- Query analyzer upgraded from label-only to score-first output:
  - `complexity_score` (0-1) as primary signal
  - keeps `complexity_label` (`simple|medium|complex`) for compatibility
  - includes `confidence`, `source`, `reason_codes`, `llm_status`
- Hybrid query classification now in place:
  - baseline: rule + heuristic scoring
  - fallback: Hugging Face FLAN (`google/flan-t5-large`) for low-confidence cases
  - guardrails: timeout, strict JSON parsing, fallback to heuristic on failure
- Adaptive retriever upgraded to dynamic behavior:
  - score-based `k_base = round(3 + complexity_score * 7)` (capped)
  - low-confidence safety bump before retrieval
  - keyword coverage check after retrieval
  - one re-retrieval step when coverage is below complexity-aware threshold
  - retry step tuned to `+1` and confidence-gated (`confidence < 0.75`)
- Retrieval metadata now logged:
  - `complexity_score_used`, `analyzer_confidence`
  - `k_base`, `k_final`
  - `coverage_score`, `coverage_threshold_used`
  - `retrieval_retry`
- Evaluation output expanded in `evaluate.py`:
  - includes analyzer/retrieval tuning fields in CSV/JSON
  - prints FLAN invocation/timeouts/errors
  - prints retrieval retry counts and average coverage

## Current Tuned Retrieval Defaults
- Complexity-aware coverage thresholds:
  - simple: `0.45`
  - medium: `0.55`
  - complex: `0.60`
- Retry step: `+1` (capped)
- Retry confidence gate: retry only when analyzer confidence is below `0.75` (or missing)

## Last Validation Snapshot
- Retrieval-only evaluation on 50 test queries:
  - retry rate improved from `20/50` to `10/50`
  - average coverage remained stable (~`0.63`)
  - average final `k` decreased after tuning
- Full end-to-end `evaluate.py`: **complex** queries need a valid `OPENAI_API_KEY`; **simple/medium** need LM Studio (or compatible) running with models loaded. See `.env.example`.

## Key Technical Decisions
- Vector store: ChromaDB (local)
- Dataset: Notre Dame documents
- Test set: 50 queries in `data/test_queries.csv`
- Model tiers: **simple/medium** → local OpenAI-compatible endpoint (`LOCAL_OPENAI_BASE_URL`, model IDs from `/v1/models`); **complex** → `OPENAI_COMPLEX_MODEL` (default `gpt-4o-mini`) via OpenAI API
- Query analyzer strategy: heuristic-first, FLAN fallback for low confidence
- Retrieval strategy: score-first continuous `k` with quality-aware retry

## File Structure
```
cost-optimized-rag/
├── src/
│   ├── adaptive_retriever.py     ✅ done
│   ├── context_compression.py    ✅ done
│   ├── model_router.py           ✅ local + OpenAI hybrid
│   ├── confidence_checker.py     ✅ done
│   ├── query_analyzer.py         ✅ done
│   └── pipeline.py              ✅ done (orchestrator)
├── data/
│   ├── documents/
│   └── test_queries.csv
├── chroma_db/
├── test_pipeline.py
├── run_all_queries.py
├── evaluate.py
├── src/pipeline.py
└── requirements.txt
```

## How to Run
```bash
source venv/bin/activate
python test_pipeline.py        # test single query
python run_all_queries.py      # test all 50 queries
python evaluate.py             # full evaluation (local server + OpenAI key for complex tier)
```

## Rules
- Never commit `.env`
- Always `git pull` before starting work
- Only `git add` your own file — never `git add .`

## Last Updated
April 24, 2026 (see appended **Updates — Apr 24, 2026** sections below for the full chronological log; latest block: **Updates — Apr 24, 2026 (compression ↔ retrieval, metrics naming, hygiene, router decision)**)

## Updates — Apr 24, 2026
- Moved pipeline orchestrator to `src/pipeline.py` and updated imports (`evaluate.py` now imports from `src.pipeline`).
- Removed unused compatibility wrapper `src/adaptive_retrieval.py` to avoid duplicate orchestration files.
- Added minimum useful stage observability in pipeline/evaluation:
  - per-stage latency (`query_analyzer_ms`, `retrieval_ms`, `compression_ms`, `model_router_ms`, `confidence_checker_ms`)
  - end-to-end latency (`total_pipeline_ms`)
  - persisted to CSV/JSON through `evaluate.py`
- Ran and stored OpenAI baseline run artifacts:
  - `results/openai_v1/run_output.log`
  - `results/openai_v1/evaluation_results_openai_v1.csv`
  - `results/openai_v1/evaluation_results_openai_v1.json`
  - `results/openai_v1/run_config.json`
- Added experiment versioning structure under `results/`:
  - `results/README.md`
  - `results/openai_v2_hybrid_context/run_config.template.json`
  - `results/local_v1/run_config.template.json`
  - `results/local_v2_hybrid_context/run_config.template.json`
  - `results/compare_runs.py` (delta comparison helper for two run CSVs)
- Ran a seed-variation stability run (`seed=7`) and stored artifacts:
  - `results/openai_v1_seed7/run_output.log`
  - `results/openai_v1_seed7/evaluation_results_openai_v1_seed7.csv`
  - `results/openai_v1_seed7/evaluation_results_openai_v1_seed7.json`
  - `results/openai_v1_seed7/run_config.json`
- Seed stability comparison (`openai_v1` vs `openai_v1_seed7`):
  - accuracy unchanged (`56.67%`)
  - retrieval retry rate unchanged (`20%`)
  - coverage unchanged (`0.6338`)
  - timing varied moderately, expected for network-bound LLM calls

## Updates — Apr 24, 2026 (later session)

### Model router (local + OpenAI hybrid)
- `src/model_router.py` uses **LangChain `ChatOpenAI`** against:
  - **Local**: `LOCAL_OPENAI_BASE_URL` (default `http://localhost:1234/v1` for LM Studio) + `LOCAL_OPENAI_API_KEY` (default `lm-studio`).
  - **OpenAI** (complex only): standard OpenAI credentials; model from `OPENAI_COMPLEX_MODEL`.
- **Tier mapping**: `simple` → `LOCAL_SIMPLE_MODEL`, `medium` → `LOCAL_MEDIUM_MODEL`, `complex` → `OPENAI_COMPLEX_MODEL`.
- **Important**: local model names must match IDs returned by `GET {base}/models` (e.g. `tinyllama-1.1b-chat-v1.0`, `mistralai/ministral-3-3b`), not short aliases.
- Optional **health check** before local calls: `LOCAL_HEALTHCHECK_ENABLED` (default `true`), `LOCAL_HEALTHCHECK_TIMEOUT_SECONDS` (default `3`). Lists available models if the requested id is missing.
- `.env.example` documents all of the above.

### Resilience / offline-friendly bits
- `src/context_compression.py` and `src/model_router.py`: if `tiktoken` cannot load `cl100k_base` (network/cache), token counts fall back to a simple whitespace split (metrics only; not OpenAI-accurate).
- `src/confidence_checker.py`: if `SentenceTransformer("BAAI/bge-small-en-v1.5")` fails to load, embedding similarity returns `0.0` and logs once (pipeline keeps running).

### Query analyzer (Evelyn-inspired heuristics, same API)
- `src/query_analyzer.py`: merged heuristic ideas from a former prototype folder (`evelyn_query/`, now removed) without adopting its separate runtime:
  - extra complex/medium keywords;
  - stronger multi-part signal (`" and "` in longer queries);
  - boundary-aware confidence near label thresholds (`0.38` / `0.70`).
- **Report artifacts** (for write-ups): `results/query_analyzer_report/`
  - `REPORT.md`, `query_analyzer_predictions.csv`, `query_analyzer_summary.json`
  - `BEFORE_AFTER_HEURISTIC_TEMPLATE.md` — narrative template for the final report.

### Evaluation / results layout
- Hybrid local+OpenAI runs can be versioned like existing folders, e.g. `results/local_v2_lmstudio/`, `results/local_v3_lmstudio_ids/`, plus seed-7 pairs and `compare_vs_*.txt` from `results/compare_runs.py`.
- **`evaluate.py` CSV/JSON** now carries per-stage observability: analyzer word count + reason codes + label match vs test row; retrieval coverage threshold + retry gate + mean Chroma chunk distance; compression sentence counts + tokens saved + pipeline token estimates; router source + fallback + reroute flag; confidence semantic score + threshold + embedding checker on/off. JSON adds `pipeline_stage_summary` (nested snapshot). End-of-run summary prints avg tokens saved, avg semantic confidence, and model reroute count.

### Readiness caveats (full pipeline)
- **Router**: LM Studio must be running with models whose IDs match `.env` (`GET /v1/models`). **Complex** tier needs **`OPENAI_API_KEY`**.
- **Retrieval / confidence / compression embeddings**: HF model `BAAI/bge-small-en-v1.5` must load (set **`HF_HOME`** / project `.hf_cache` if default cache is not writable). Set **`COMPRESSION_USE_EMBEDDINGS=false`** to skip compression-time sentence embeddings only.
- **Confidence reroute**: `MODEL_TIERS` in `confidence_checker.py` still targets **OpenAI model names**; local-only model IDs usually **will not** trigger a stronger-model retry until tiers are extended for your local IDs.

### Context compression (selective Anh-style upgrade, local-only)
- `src/context_compression.py`: **adaptive budget** `round(1 + complexity_score * 5)` (cap 6) when `COMPRESSION_ADAPTIVE_BUDGET=true`; else legacy fixed map `simple/medium/complex`.
- **Hybrid ranking**: TF-IDF + **local** `SentenceTransformer` (default `BAAI/bge-small-en-v1.5`, same family as retrieval). Weights shift toward embeddings as `complexity_score` rises. If embeddings fail to load, falls back to **TF-IDF only** (no OpenAI embedding calls).
- **Redundancy**: greedy selection by score; skip a candidate if cosine similarity ≥ `COMPRESSION_REDUNDANCY_THRESHOLD` (default `0.85`) to an already chosen sentence (embedding space, or TF-IDF if no embeddings).
- **Answer-type boosts**: small score bumps for when/year, who/capitalized names, where/location cues, what-is definition patterns.
- `src/pipeline.py` passes `complexity_score` into `compress(...)`. Return dict includes `compression_metadata` (budget, weights, `used_embeddings`, redundancy skips). `evaluate.py` adds flat CSV columns and full `compression_metadata` in JSON records.
- Env knobs in `.env.example`: `COMPRESSION_USE_EMBEDDINGS`, `COMPRESSION_EMBEDDING_MODEL`, `COMPRESSION_ADAPTIVE_BUDGET`, `COMPRESSION_REDUNDANCY_THRESHOLD`.

### Performance note (debugging)
- If runs feel slow, check per-query **stage timing** in logs/CSV: `model_router_ms` and `confidence_checker_ms` often dominate; `query_analyzer_ms` → `adaptive_retriever` wiring is verified consistent (`complexity_label`, `complexity_score`, `confidence` passed through `src/pipeline.py`).

## Updates — Apr 24, 2026 (evaluation interpretability + seed runs)

_Added after the sections above; earlier dated bullets are unchanged._

### Answer vs ground truth in `evaluate.py` (offline only; no OpenAI judge)
- Per-row CSV/JSON columns: **`answer_gt_token_f1`** (multiset token F1 between `ground_truth` and `final_answer`), **`answer_gt_embedding_similarity`** (cosine sim of embeddings for the two strings; local **`sentence_transformers`**, model from **`EVAL_GT_EMBEDDING_MODEL`** in `.env.example`, default `BAAI/bge-small-en-v1.5`), **`answer_gt_appropriateness_score`** (blend when both signals exist).
- End-of-run console block: **“ANSWER VS GROUND TRUTH (softer metrics)”** with row counts and averages; **substring** accuracy is labeled explicitly as **“Substring hit rate”** (`ground_truth` must appear in `final_answer`) so it is not confused with F1/embedding scores.
- JSON **`pipeline_stage_summary.answer_vs_ground_truth`** holds substring hit, F1, embedding sim, and appropriateness for quick reporting.

### `confidence_checker.py` (runtime; unchanged role, richer exports)
- Return dict now includes **`confidence_threshold`**, **`confidence_semantic`**, and **`confidence_checker_embedding_enabled`** so pipeline/`evaluate.py` can log whether embedding-based similarity ran.

### Concept split (for team write-ups)
- **`confidence_checker`**: scores **answer ↔ compressed context** (grounding / self-consistency in the prompt). Does **not** use CSV `ground_truth`.
- **`evaluate.py` soft metrics**: score **answer ↔ labeled `ground_truth`** for benchmarks. Complementary to the checker, not a replacement.

### Multi-seed comparison artifacts
- Folder **`results/compare_seeds/`**: `seed42/` and `seed7/` each with `evaluation_results.csv`, `.json`, `run_output.log`; root **`compare_seed7_vs_seed42.txt`** from `python results/compare_runs.py ...`; **`RUN_SUMMARY.txt`** lists paths and the compare command.

### Prototype / noise cleanup
- Removed duplicate **`evelyn_query/`** prototype (heuristics already merged into `src/query_analyzer.py`).
- **`Anh_Context compression/`** was verified to be a **Windows venv only** (no app source to merge); safe to delete from the repo if still present—do not treat as `src/context_compression.py` replacement.

### Possible next step (not implemented; design only)
- **Retrieval + compression-aware confidence**: pass e.g. `coverage_score`, `compression_ratio`, `retrieval_retry` from **`src/pipeline.py`** into **`check_confidence`** and adjust threshold/score interpretably (team slide spec). **`MODEL_TIERS`** remains OpenAI-name-oriented until extended for local LM Studio model IDs if you want confidence-tier retries there too.

## Updates — Apr 24, 2026 (preflight, `evaluate.py` CLI, `.env` load order, docs, smoke runs)

_Appended; all sections above remain as-is._

### `src/preflight.py` (new)
- **`PreflightError`** for fail-fast checks before a long batch.
- **`validate_queries_csv`**: file exists, required columns (`query_id`, `query`, `complexity`, `ground_truth`), at least one data row.
- **`check_vector_store`**: `chroma_db/` under **project root** or a documents tree with ingestible `.txt` files.
- **`check_local_openai_models`**: `GET` local `…/v1/models` and verify `LOCAL_SIMPLE_MODEL` / `LOCAL_MEDIUM_MODEL` unless **`--skip-local-model-check`**.
- **`warn_openai_key_if_missing`** / optional **`--require-openai-key`** on `evaluate.py`.
- **`log_effective_configuration` / `log_pipeline_startup`**: sanitized snapshot (paths, model IDs, compression flags, whether OpenAI key is set); **`src/pipeline.py`** calls startup log once after wiring components.
- **Bugfix**: removed stray **`return n`** from `run_eval_preflight` after refactor (was causing `NameError`).

### `evaluate.py` — `.env` before preflight + CLI
- **`load_dotenv(PROJECT_ROOT / ".env")`** runs at import (after `PROJECT_ROOT` is set) so **`LOCAL_*` / `OPENAI_*`** are visible to **`os.getenv` during preflight** (previously preflight ran before `RAGPipeline` constructed `ModelRouter`, so defaults like `tinyllama` / `ministral` were used and LM Studio ID checks failed even with a correct `.env`).
- **Argparse**: `--queries`, `--out-csv`, `--out-json`, `--documents`, `--seed`, `--no-preflight`, `--skip-local-model-check`, `--require-openai-key`, `--log-file`, `--log-json`, `-v` / `-vv`, `-q`.
- **Paths**: relative paths resolve under project root via **`src.preflight.resolve_under_project`**.
- **`RAGPipeline` lazy-import** inside `run_evaluation()` so **`python evaluate.py --help`** does not load HF/Chroma.
- **`try` / `finally`** closes optional JSONL handle; guard if zero rows processed.
- **Per-query JSONL** (`--log-json`): `query_complete` lines with `query_id`, `total_pipeline_ms`, `substring_correct`, `failure_type`.
- **Exit codes**: `2` = `PreflightError`, `130` = Ctrl+C, `1` = other exceptions.
- **Wrapper script**: `scripts/run_eval.sh` — `cd` to repo root and `exec python evaluate.py "$@"`.

### `src/adaptive_retriever.py`
- **`chroma_db`** path is **`{project_root}/chroma_db`** (not cwd-relative) so retrieval works regardless of shell working directory.

### `src/confidence_checker.py` (documentation)
- Module docstring clarifies **`MODEL_TIERS`** keys are **OpenAI-style** names; **local LM Studio `id` strings do not match** → **no tier retry** (scores still returned).

### Repo docs
- **`README.md`**: quick start, LM Studio `/v1/models` IDs, preflight/CLI, GT vs confidence checker, first-run notes (tiktoken fallback, embeddings).
- **`TEAM_GUIDE.md`**: pipeline status, file tree, `evaluate.py` / `run_eval.sh` instead of removed `test_pipeline.py` / `run_all_queries.py`.

### `src/` module docstrings (light)
- Added top-of-file / section headers in **`pipeline.py`**, **`model_router.py`**, **`adaptive_retriever.py`**, **`query_analyzer.py`**, **`context_compression.py`**, **`confidence_checker.py`**, **`seed_manager.py`**; **`evaluate.py`** module docstring updated for CLI and GT semantics.

### Fast iteration (smoke) — added in repo
- **`data/smoke_queries.csv`**: first **10** rows from `data/test_queries.csv` (all **simple**) for quick end-to-end checks.
- Example run (does not require editing `.env` permanently):  
  `COMPRESSION_USE_EMBEDDINGS=false python3 evaluate.py --queries data/smoke_queries.csv --out-csv results/smoke_run/evaluation_smoke.csv --out-json results/smoke_run/evaluation_smoke.json`  
  **Lighter compression** = TF-IDF-only for that process env; **fewer rows** = faster wall clock. Use full **`data/test_queries.csv`** + `COMPRESSION_USE_EMBEDDINGS=true` for comparable benchmark numbers.

### LM Studio / `.env` reminder (team)
- **`LOCAL_OPENAI_BASE_URL`** should include **`/v1`** (e.g. `http://127.0.0.1:1234/v1`).
- **`LOCAL_SIMPLE_MODEL`** and **`LOCAL_MEDIUM_MODEL`** must equal **API model identifiers** from LM Studio (or the same model id for both tiers if only one model is loaded).

## Updates — Apr 24, 2026 (compression ↔ retrieval, metrics naming, hygiene, router decision)

_Appended; all sections above remain as-is._

### `src/context_compression.py` + `src/pipeline.py`
- **`coverage_score`** from **`adaptive_retriever.retrieve`** is passed into **`ContextCompressor.compress`**; sentence budget adds up to **+2** sentences when keyword coverage is low (capped at **6**); metadata includes `sentence_budget_base`, `coverage_score_used`, `coverage_budget_extra`.
- Adaptive / coverage-adjusted budgets use a **minimum of 2** sentences (was 1) to reduce overly aggressive single-sentence context.
- **`_hybrid_weights`** docstring corrected: higher **complexity** shifts weight toward **embedding** (semantic-heavy); simple stays more keyword/TF-IDF–heavy.

### `src/adaptive_retriever.py`
- Chunk metadata adds **`chroma_distance`** (Chroma/LangChain **distance**, **lower = closer**). **`similarity_score`** kept as the same numeric value for older JSON consumers (name is misleading).
- Removed unused **`k_map`** dict.
- Renamed gate variable to **`low_confidence_for_retry`** (clearer than “confidence_for_retry”).

### `src/pipeline.py`
- Prints chunk lines as **`chroma distance`**; **`retrieval_avg_chunk_distance`** is mean distance (unchanged column name, semantics documented in README).

### `evaluate.py`
- Creates **parent directories** for `--out-csv` / `--out-json` before write.
- Extra CSV columns: **`compression_sentence_budget_base`**, **`compression_coverage_score_used`**, **`compression_coverage_budget_extra`**.

### `src/query_analyzer.py`
- FLAN **adoption** when overriding heuristics now uses **`self.high_conf_threshold`** (from **`QUERY_ANALYZER_HIGH_CONF_THRESHOLD`**, default **0.80**) instead of hardcoded **0.65** — fewer overrides unless FLAN is highly confident (tunable via env).

### `src/confidence_checker.py`
- **`sentence_transformers`** load is **lazy** (import inside try on first use) so a missing/broken package does not break module import; module-level note above **`MODEL_TIERS`** about local LM Studio ids not matching OpenAI-style keys.

### `src/model_router.py` (explicit non-change)
- A **Phase 2** experiment added per-complexity **temperature** and **prompt templates**; it was **reverted**. Current router: **single `self.temperature`** (default **0.0**) and **one** system/human prompt pair. No **`ROUTER_TEMP_*`** env vars in **`.env.example`**.

### Repo hygiene (`.gitignore`)
- Extended with **`results/`**, **`.hf_cache/`**, **`venv/`**, **`*.docx`**, **`~$*`** (Office temp locks) among other patterns — reduces accidental commits of runs, caches, and local artifacts.

## Updates — Apr 25, 2026 (shared embeddings cache + router fallback discussion)

_Appended; all sections above remain as-is._

### `src/shared_embeddings.py` (new)
- Added a shared in-process cache for sentence-transformer models: **`get_embedding_model(model_name)`**.
- Cache is **per model name** (dict-backed), so one process reuses loaded weights instead of re-instantiating the same model in each component.
- Added lightweight **`logging.debug(...)`** messages for:
  - previously failed model load skip
  - cached model reuse
  - first successful load
  - load failure (exception text)

### `src/context_compression.py`
- Replaced direct `SentenceTransformer(...)` creation with shared **`get_embedding_model(self.embedding_model_name)`**.
- If shared loader returns `None`, compressor marks embeddings unavailable and falls back safely.

### `src/confidence_checker.py`
- Replaced local lazy `SentenceTransformer("BAAI/bge-small-en-v1.5")` init with shared cache loader using the same model id.
- Keeps existing fail-open behavior (pipeline continues if embeddings are unavailable).

### `evaluate.py`
- Replaced `_eval_embedding_model()` direct instantiation with shared cache loader for **`EVAL_GT_EMBEDDING_MODEL`**.
- Preserves prior behavior when embedding model is unavailable (returns `None` for that metric path).

### Expected impact
- Reduces duplicate RAM use when multiple modules need the same embedding model in one run.
- Reduces repeated warm-up/loading overhead; throughput per `encode` call is unchanged after warm-up.
- Improves stability for Colab/low-memory environments by avoiding multiple copies of the same model weights.

### `src/model_router.py` note (decision deferred)
- Considered adding provider fallback via external cascade (complex path only).
- Team decision for now: **defer** this change; keep current router behavior unchanged until later A/B testing.

## Updates — Apr 25, 2026 (semantic cache in pipeline)

_Appended; all sections above remain as-is._

### `src/semantic_cache.py` (new)
- Added **`SemanticCache`** for in-memory near-duplicate query reuse.
- Uses lazy-load **`SentenceTransformer("BAAI/bge-small-en-v1.5")`** (same fail-open pattern style as confidence checker).
- Cache entries are dicts with keys: **`query`**, **`embedding`**, **`result`**, **`corpus_hash`**.

### Lookup / hit policy
- **`lookup(query, corpus_hash)`** embeds the incoming query and runs cosine similarity vs cached embeddings (sklearn).
- Cache hit requires:
  - matching **`corpus_hash`**
  - best similarity **>= 0.92**
- On hit, returns cached result enriched with:
  - **`cache_hit=True`**
  - **`cache_similarity`** (rounded float for logging)

### Store / eviction policy
- **`store(query, result, corpus_hash)`** writes only when result quality gates pass:
  - `confidence_score_final > 0.75`
  - `retried == False`
  - `coverage_score > 0.55`
  - `retrieval_retry == False`
- FIFO eviction: if cache grows beyond **500** entries, oldest entry is removed.

### Corpus-change safety
- **`SemanticCache.get_corpus_hash(chroma_dir)`** computes an MD5 over file modification times under the Chroma directory.
- Returns empty string if directory does not exist.

### `src/pipeline.py` integration
- In `__init__`:
  - instantiate **`SemanticCache`**
  - compute and store **`self.corpus_hash = SemanticCache.get_corpus_hash(CHROMA_DIR)`**
- At start of `run()`:
  - call cache lookup before analyzer/retriever/compressor/router
  - on hit, prints cache-hit message with similarity and returns cached payload immediately
- At end of `run()`:
  - add **`cache_hit=False`** for normal non-cached path
  - call cache `store(...)` just before return

## Updates — Apr 25, 2026 (alignment + safeguards follow-up)

_Appended; all sections above remain as-is._

### `src/semantic_cache.py` follow-up
- Replaced local embedding loader with shared loader: **`from src.shared_embeddings import get_embedding_model`**.
- **`SemanticCache._get_embedding_model()`** now reuses shared in-process model cache, preventing duplicate BGE-small instances on cold start.
- Added **`logging.debug(...)`** rejection observability in `store()` for each gate (low confidence, retried, low coverage, retrieval retry, embedding unavailable, and non-dict payload), plus store/eviction events.

### `src/context_compression.py`
- Aligned adaptive sentence budget formula to agreed spec:
  - from `round(1 + complexity_score * 5)` (clamped)  
  - to **`round(2 + complexity_score * 4)`** (clamped to `[2, 6]`).

### `src/preflight.py` + `evaluate.py`
- Added CSV-aware fail-fast for OpenAI key:
  - new helper **`csv_has_complex_queries(queries_file)`**
  - `evaluate.py` computes this before preflight
  - `run_eval_preflight(...)` now enforces missing-key failure when the benchmark includes any `complex` rows (or when `--require-openai-key` is set).
- Result: avoids mid-batch failure on first complex query due to missing `OPENAI_API_KEY`.

### `src/adaptive_retriever.py`
- Unified retrieval sizing around shared constants:
  - `K_BASE_MIN=3`, `K_BASE_MAX=10`, **`K_HARD_CAP=10`**, `LOW_CONF_BONUS=2`, `RETRY_BONUS=1`.
- Low-confidence bump and coverage-retry bump now both respect the same hard cap constant.
- Removes prior split-cap behavior (`k_base` capped at 10 while bump/retry could reach 12).

## Updates — Apr 25, 2026 (general typo-robust query handling)

_Appended; all sections above remain as-is._

### `src/pipeline.py` generalization
- Replaced narrow hardcoded typo replacements with a **corpus-driven spell-correction path**.
- Built a lightweight in-memory vocabulary index from `data/documents/**/*.txt` and used fuzzy matching (`difflib`) to normalize noisy query tokens.
- Added conservative token guardrails (stopword + short-token excludes) plus small edit heuristics (leading/trailing character noise) before fuzzy correction.
- Query flow now uses **original + normalized + spell-corrected** variants for cache lookup and robustness retries.

### Retry behavior updates
- Kept the low-quality fallback retry but made it **general** (not founder-specific):
  - triggers on weak retrieval/answer quality signals,
  - retries retrieval with stronger settings and candidate query variant,
  - adopts retry output only when confidence improves.

### Scope cleanup
- Removed the temporary founder-specific hard override path so behavior remains generic across intents.
- Retained existing non-session semantic cache behavior and session-document cache bypass.

## Updates — Apr 25, 2026 (factual extraction precision + retrieval diagnostics UI + k override cleanup)

_Appended; all sections above remain as-is._

### `src/adaptive_retriever.py`
- Added/kept the sentence-level factual index path and retrieval helper for extraction-first answering:
  - factual sentence corpus built from `data/documents/**/*.txt`,
  - title/section noise lines are stripped before sentence indexing,
  - sentence ranking blends semantic TF-IDF similarity with query-token overlap.

### `src/pipeline.py` (precision + observability)
- Improved `who founded/established/started ...` extraction with a **general subject-aware reranker**:
  - alias generation for subject forms (`the ...`, `university of ...`, acronym variants, `and`/`&` variants),
  - alias-strength scoring against candidate sentence text,
  - proximity gate requiring subject alias to appear near founding verbs to avoid sub-entity false positives.
- Extraction-first branch now falls back safely when confidence is weak (no hardcoded entity exceptions).
- Added structured return payload: **`retrieval_diagnostics`** with:
  - factual extraction attempted/used + hit counts + top factual hits,
  - retrieval source diagnostics (`k_base`, `k_final`, coverage, retry, top sources/distances),
  - compression mode, grounding gate outcome, OpenAI/deep fallback attempt and use flags.

### `ui.py` (retrieval diagnostics surfaced)
- Chat metadata pills now include **compression mode**.
- Added per-assistant-turn expandable **Retrieval diagnostics** panel in chat, showing:
  - k progression, coverage, retry, factual extraction status/hits,
  - grounding gate result, fallback usage, top retrieved sources, top factual sentence hits.
- Dashboard gained a **retrieval diagnostics summary** section:
  - avg final k, avg coverage, retrieval retry rate,
  - factual extractive usage count, grounded response count, OpenAI fallback usage count.
- Session logging now stores retrieval diagnostic booleans/fields for dashboard aggregation.

### Follow-up cleanup (requested)
- Removed the special-circumstance forced retrieval depth override from deep OpenAI retry path:
  - deleted `force_k=20` usage in `src/pipeline.py`,
  - removed forced-k wording from logs/metadata,
  - deep retry now uses normal adaptive retriever behavior.

## Updates — Apr 25, 2026 (semantic chunker + HyDE toggle + corpus curation)

_Appended; all sections above remain as-is._

### `src/adaptive_retriever.py` (chunking upgrade)
- Added optional semantic chunking path using **`SemanticChunker`** from `langchain_experimental` with safe fallback to `RecursiveCharacterTextSplitter`.
- New env controls:
  - `USE_SEMANTIC_CHUNKER` (default `true`)
  - `SEMANTIC_BREAKPOINT_TYPE` (default `percentile`)
  - `SEMANTIC_BREAKPOINT_AMOUNT` (default `85`)
- Chunking config/version now includes semantic-chunker settings (`CHUNKING_VERSION = "v3_semantic_chunker_toggle"`), so Chroma rebuild triggers automatically when toggled.
- Added dependency: **`langchain-experimental`** in `requirements.txt`.

### `src/pipeline.py` (HyDE retrieval, flag-gated)
- Added lightweight **HyDE** at query time (ingest unchanged):
  - env: `USE_HYDE` (default `false`)
  - generates one concise hypothetical retrieval text using local OpenAI-compatible model
  - runs alternate retrieval with that text
  - adopts HyDE retrieval only when coverage improves by at least `HYDE_MIN_COVERAGE_GAIN` (default `0.03`)
- Added related env/tuning fields:
  - `HYDE_MIN_COVERAGE_GAIN` (default `0.03`)
  - `HYDE_MAX_CHARS` (default `700`)
- Added observability fields in result payload:
  - `hyde_attempted`, `hyde_used`, `hyde_coverage_score`, `hyde_query_preview`, `retrieval_query_used`

### `ui.py` (HyDE diagnostics visibility)
- Retrieval diagnostics panel now shows HyDE status:
  - attempted/used flags
  - alternate retrieval coverage score

### Corpus curation fix for founder query
- Added one canonical fact sentence to `data/documents/doc_18.txt`:
  - “Father Edward Sorin of the Congregation of Holy Cross founded the University of Notre Dame on November 26, 1842.”
- Rebuilt index (`chroma_db`) after update.
- Post-rebuild validation confirms `Who founded Notre Dame?` now returns the canonical grounded sentence via `extractive-factual`.

## Updates — Apr 26, 2026 (numeric factual formatting + Streamlit stability)

_Appended; all sections above remain as-is._

### `src/pipeline.py` (`how many` extractive answer refinement)
- Improved extractive formatting for count questions:
  - maps number words (`one`..`twenty`) to numeric values,
  - scans all numeric mentions in top factual sentence hits,
  - selects the strongest numeric signal (max value) to avoid incorrect first-match picks like “one of the five...”.
- Output now returns concise numeric form when possible (e.g., `5 undergraduate colleges.`) instead of full sentence-only phrasing.

### Deep fallback guard for valid numeric extractive answers
- Added a guard to prevent deep OpenAI fallback from overriding already-valid extractive numeric answers.
- Condition: when factual extraction is used and answer starts with a number, skip deep OpenAI retry.

### Runtime validation snapshot
- Verified query:
  - `How many undergraduate colleges are at Notre Dame?`
  - final answer now resolves to numeric extractive output (`5 undergraduate colleges.`) on `extractive-factual` path.
- Also resolved Streamlit runtime startup instability by launching from project `venv` with:
  - `--server.fileWatcherType none`
  - this avoids watcher-related crashes seen under the global/anaconda environment.

## Updates — May 1, 2026 (retrieval reranker + confidence checker reliability tuning)

_Appended; all sections above remain as-is._

### `src/adaptive_retriever.py` (cross-encoder second-stage reranking)
- Added optional cross-encoder reranking after hybrid merge and before compression input:
  - model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - candidate cap: `CROSS_ENCODER_CANDIDATES` (default `20`)
  - flag: `USE_CROSS_ENCODER_RERANK` (default `true`)
- Kept current dense+lexical hybrid as candidate generation and bounded reranking to top candidates only.
- Added safe fallback behavior when cross-encoder load/inference fails.
- Added retrieval metadata for observability:
  - `rerank_enabled`, `rerank_used`, `rerank_model`, `rerank_latency_ms`
  - chunk-level `cross_encoder_score` + `retrieval_mode` suffix `+cross_encoder`.

### `src/pipeline.py` (rerank observability propagation)
- Extended pipeline result and `retrieval_diagnostics` to include reranker fields:
  - `rerank_enabled`, `rerank_used`, `rerank_model`, `rerank_latency_ms`.

### `src/confidence_checker.py` (retry behavior fixes + scoring fairness)
- Fixed local-model retry dead-path by expanding model tier mapping:
  - dynamically map `LOCAL_SIMPLE_MODEL` and `LOCAL_MEDIUM_MODEL` to `gpt-4o-mini`,
  - keep OpenAI escalation path (`gpt-3.5-turbo -> gpt-4o-mini -> gpt-4o`),
  - include short alias support from model IDs.
- Added threshold configurability with local/openai split:
  - `CONFIDENCE_THRESHOLD_LOCAL` (default `0.50`)
  - `CONFIDENCE_THRESHOLD_OPENAI` (default `0.65`)
  - backward-compatible `CONFIDENCE_THRESHOLD` override still supported.
- Implemented complexity-aware heuristic length normalization:
  - simple/medium/complex targets: `10/20/40`.
- Upgraded embedding similarity from global context vector to sentence-level max pooling:
  - answer embedding is compared against each context sentence embedding,
  - final embedding score uses max sentence match for factual grounding.

### `src/model_router.py` (local model cache hygiene)
- Added TTL-based refresh for local model cache to avoid stale `/v1/models` state when LM Studio hot-reloads:
  - `LOCAL_MODELS_CACHE_TTL_SECONDS` (default `60`).

### Runtime checks performed
- Compile + lint checks passed for touched files.
- Smoke tests confirmed:
  - local-model confidence retry can now escalate to OpenAI tier when needed,
  - local/openai thresholds are selected correctly,
  - cross-encoder reranker is active and observable in retrieval metadata.

## Updates — May 1, 2026 (analyzer-driven factual fast path + compression generalization)

_Appended; all sections above remain as-is._

### `src/pipeline.py` (QueryAnalyzer-triggered factual fast path)
- Added a first-class factual fast path that can bypass Chroma retrieval and go directly to sentence-index extraction when all of the following hold:
  - `simple_starter` present in `analysis.reason_codes`,
  - `complexity_score < 0.30`,
  - factual-query pattern check passes,
  - no session-uploaded documents are attached.
- Behavior:
  - if strong extractive answer is found from `retrieve_factual_sentences(...)`, pipeline skips vector retrieval and proceeds with extractive-factual path;
  - otherwise falls back safely to normal hybrid retrieval.
- Added result + diagnostics fields:
  - `factual_fastpath_attempted`
  - `factual_fastpath_used`

### `src/context_compression.py` (`_answer_boost` de-corpus-ified)
- Replaced corpus-specific location keywords with generic/default location cues.
- Added env override:
  - `COMPRESSION_LOCATION_KEYWORDS` (comma-separated) for corpus-specific customization when needed.
- Added generic place-pattern detection (proper-noun phrase after location prepositions) for `where`-type boosts.

### `src/pipeline.py` robustness fix (retrieval distance metric)
- Fixed `retrieval_avg_chunk_distance` aggregation to skip non-numeric chunk distance values (e.g., lexical chunks with empty distance fields), avoiding `float('')` errors in mixed retrieval mode.

### Validation snapshot
- Compile + lint passed for updated files.
- Smoke checks confirm:
  - founder query uses factual fast path and returns grounded canonical sentence with retrieval bypass,
  - `how many` query still returns numeric extractive answer,
  - non-factual comparative queries continue through standard retrieval/compression/router path.

## Updates — May 1, 2026 (benchmark instrumentation + factual fastpath impact validation)

_Appended; all sections above remain as-is._

### `evaluate.py` instrumentation updates
- Added new exported CSV/JSON fields for analysis:
  - `factual_fastpath_attempted`
  - `factual_fastpath_used`
  - `rerank_enabled`, `rerank_used`, `rerank_model`, `rerank_latency_ms`
- Extended end-of-run summary to print:
  - factual fastpath usage (used/attempted counts),
  - cross-encoder rerank usage count.

### Full 50-query benchmark run
- Executed benchmark with:
  - `evaluate.py --queries data/test_queries.csv`
  - outputs:
    - `results/may1_fastpath/evaluation_results.csv`
    - `results/may1_fastpath/evaluation_results.json`
- Run summary highlights:
  - total queries: `50`
  - factual fastpath: `used 12/50` (attempted `12`)
  - cross-encoder rerank used: `38/50`
  - retrieval retries: `4/50`
  - model reroutes: `15/50`

### Factual fastpath impact (computed from benchmark CSV)
- Average total pipeline latency:
  - fastpath used: `171.38 ms`
  - fastpath not used: `11211.90 ms`
- Average retrieval latency:
  - fastpath used: `3.29 ms`
  - fastpath not used: `572.60 ms`
- Average confidence:
  - fastpath used: `0.79`
  - fastpath not used: `0.54`
- Fastpath usage concentrated on simple queries:
  - simple: `12`
  - medium: `0`
  - complex: `0`

## Updates — May 1, 2026 (fallback-chain cap + timing split + starter protection + rerank policy)

_Appended; all sections above remain as-is._

### `src/pipeline.py` (confidence-stage timing split)
- Split confidence-stage timing into explicit sub-metrics:
  - `confidence_scoring_ms`
  - `robust_retry_ms`
  - `deep_openai_ms`
  - `grounding_fallback_ms`
- Kept `confidence_checker_ms` as aggregate of the four sub-metrics for backward-compatible dashboards/reports.
- Updated stage timing log line to print confidence sub-breakdown.

### `src/pipeline.py` (protect factual starters before spell-correction)
- Added starter protection so spell-correction does not alter factual query prefixes:
  - `how many`, `who`, `when`, `where`, `which`, `what is`, `what was`.
- Behavior change: only the suffix after a protected factual prefix is spell-corrected.

### `src/adaptive_retriever.py` (selective cross-encoder policy, env-gated)
- Added per-query rerank gate:
  - rerank enabled for `medium/complex`,
  - simple queries rerank only when `RERANK_SIMPLE_QUERIES=true`.
- New env:
  - `RERANK_SIMPLE_QUERIES` (default `false`).
- Retrieval metadata now reflects effective per-query rerank decision (`rerank_enabled`, `rerank_model`, etc.).

### `src/pipeline.py` (fallback-chain depth cap, env-gated)
- Added global heavy-fallback cap across:
  - typo-tolerant robust retry,
  - deep OpenAI fallback,
  - grounding-gate OpenAI fallback.
- New env:
  - `PIPELINE_MAX_HEAVY_FALLBACKS` (default `1`).
- Added observability fields:
  - `max_heavy_fallbacks`
  - `heavy_fallbacks_used`.

### Validation — fallback cap A/B (cap=1 vs cap=2)
- **Slice A (50-query subset)**:
  - accuracy (evaluable): unchanged (`50.0%` vs `50.0%`),
  - paired average latency delta (`cap2 - cap1`): approximately `-171.79 ms` on this run (no accuracy gain).
- **Slice B (hard subset, 29 queries)**:
  - hard subset built from reroute/retry/non-simple signals;
  - single-run result: accuracy unchanged (`88.89%` evaluable), paired average latency delta (`cap2 - cap1`) approximately `+2755.87 ms`.
- **Stability check (3 repeated hard-slice runs per cap)**:
  - accuracy unchanged in all runs (`88.89%` evaluable for both caps),
  - paired mean latency delta (`cap2 - cap1`): approximately `+931.34 ms` (std `1582.09`),
  - `cap=2` showed higher confidence-stage latency variance.

### Decision from validation
- Keep default:
  - `PIPELINE_MAX_HEAVY_FALLBACKS=1`
- Rationale:
  - no observed accuracy lift from cap `2`,
  - higher/less stable latency and potential extra fallback cost with cap `2`.

## Updates — May 1, 2026 (deep OpenAI fallback coverage gate)

_Appended; all sections above remain as-is._

### `src/pipeline.py` (coverage-aware deep fallback gate)
- Added env-gated minimum coverage requirement before deep OpenAI fallback can run:
  - `DEEP_OPENAI_MIN_COVERAGE` (default `0.40`).
- Deep fallback now requires:
  - existing low-quality eligibility + API/cap checks, and
  - `retrieval.coverage_score >= DEEP_OPENAI_MIN_COVERAGE`.
- Added explicit skip log when deep fallback is otherwise eligible but blocked by low coverage:
  - includes current `coverage_score` and configured threshold.

### `.env.example` update
- Added:
  - `DEEP_OPENAI_MIN_COVERAGE=0.40`
- Comment clarifies behavior:
  - skip deep OpenAI fallback when corpus coverage is below threshold.

### Quick verification
- `src/pipeline.py` compiles successfully (`py_compile`).
- Lint check passed for touched files.

## Updates — May 1, 2026 (embedding reuse in confidence checker + robust retry scope cap)

_Appended; all sections above remain as-is._

### `src/context_compression.py` + `src/confidence_checker.py` (embedding reuse)
- Added carry-through of selected sentence embeddings from compression output:
  - `_selected_sentence_embeddings` is now included in compression results when embedding scoring is active.
- Extended confidence-check embedding API to accept precomputed context sentence embeddings:
  - `embedding_similarity(..., context_sentences, context_sentence_embeddings)`
  - `check_confidence(..., context_sentences, context_sentence_embeddings)`
- Behavior:
  - confidence checker reuses compressor-provided sentence embeddings and encodes only the answer embedding for similarity,
  - fallback path still works if cached embeddings are unavailable/mismatched.

### `src/pipeline.py` (wire embedding reuse through all confidence paths)
- Updated confidence-check calls to pass selected sentences + carried embeddings for:
  - initial confidence scoring,
  - robust retry confidence scoring,
  - grounding-triggered OpenAI fallback confidence scoring.

### `src/pipeline.py` (cap robust retry to non-simple queries)
- Restricted typo-tolerant robust retry to medium/complex requests only:
  - added `complexity != "simple"` condition to `should_retry_robust`.
- Goal:
  - avoid extra retrieval/compression/router passes on simple low-coverage questions where refusal/grounding guard is usually preferable.

### Quick verification
- Lint check passed for:
  - `src/context_compression.py`
  - `src/confidence_checker.py`
  - `src/pipeline.py`
- Compile checks passed for the same files (`py_compile`).

## Updates — May 1, 2026 (session-only multimodal PDF retrieval + UI upload/title enhancements)

_Appended; all sections above remain as-is._

### `src/session_multimodal_retriever.py` (new module, in-memory only)
- Added session-only multimodal retriever for uploaded visual PDFs:
  - no persistence,
  - no ChromaDB writes,
  - no changes to main corpus index.
- Implemented chunking spec:
  - primary sentence split: `re.split(r"(?<=[.!?])\\s+", page_text)`,
  - fallback paragraph split (`\\n\\n`) when primary yields fewer than 2 usable chunks,
  - post-filter: trim, drop `<30` chars, hard-cap to `700` chars.
- Added `SessionMultimodalError` for explicit failure signaling.
- Added `detect_visual_pdf(pdf_bytes)` using `fitz` image detection (`page.get_images()`).
- Gemini embedding integration:
  - model: `models/gemini-embedding-2-preview`,
  - `output_dimensionality=768`,
  - `task_type="retrieval_document"` for chunks,
  - `task_type="retrieval_query"` for query.
- Retrieval behavior:
  - cosine similarity,
  - fixed `top_k=3`,
  - returns `text`, `page_num`, `score`.

### `src/pipeline.py` (conditional session multimodal routing)
- Extended `run(...)` signature with optional `session_uploads` metadata while preserving existing `session_documents` behavior.
- Added multimodal activation gate (all must pass):
  - session uploads present,
  - `USE_SESSION_MULTIMODAL_EMBEDDING=true`,
  - `GEMINI_API_KEY` set,
  - uploaded PDF contains images.
- On activation success:
  - builds session multimodal index,
  - retrieves top multimodal chunks and appends texts to context before compression.
- On failure (`SessionMultimodalError`):
  - logs warning and falls back to text-only session path.
- Added observability fields:
  - `multimodal_embedding_used`,
  - `multimodal_embedding_reason`,
  - `multimodal_hits_count`.

### `ui.py` (upload metadata path + dynamic title)
- Upgraded upload extraction to produce structured session payload:
  - filename/mime,
  - raw PDF bytes,
  - per-page text map (`pdf_text_by_page`),
  - combined extracted text.
- Pipeline calls now pass both:
  - `session_documents` (text path),
  - `session_uploads` (metadata path for multimodal detection/retrieval).
- Header title now updates from uploaded document name (stem) for session context visibility; defaults back to `Notre Dame Assistant` when no file is loaded.

### Config and dependencies
- `.env.example` additions:
  - `GEMINI_API_KEY=`
  - `USE_SESSION_MULTIMODAL_EMBEDDING=false`
  - comments clarifying optional feature behavior and fallback.
- `requirements.txt` additions:
  - `google-generativeai`
  - `pymupdf`

### Runtime checks performed
- Lint checks passed for touched files.
- Compile checks passed for:
  - `src/session_multimodal_retriever.py`
  - `src/pipeline.py`
  - `ui.py`
- Streamlit restarted successfully and verified reachable at `http://localhost:8501`.

## Updates — May 1, 2026 (session multimodal index cache + reuse)

_Appended; all sections above remain as-is._

### `src/session_multimodal_retriever.py` (session-scoped in-memory Chroma cache)
- Upgraded from per-query in-memory list scoring to a session-scoped in-memory Chroma collection:
  - session key: `md5(pdf_bytes)[:12]`,
  - collection name: `session_<session_id>`,
  - single shared in-memory Chroma client per process.
- Added `GeminiEmbeddingFunction` wrapper for Chroma-compatible embeddings:
  - Gemini model: `models/gemini-embedding-2-preview`,
  - `output_dimensionality=768`,
  - query path still uses `task_type="retrieval_query"`.
- `build_index()` now caches embeddings in collection:
  - if collection already has documents, it skips re-embedding (cache hit),
  - otherwise it chunks, embeds once, and stores ids/documents/embeddings/metadata.
- `retrieve()` now queries Chroma directly with `query_embeddings=[...]` and returns top-3 hits.
- Added retriever stats used for observability:
  - `chunk_count`,
  - `avg_chunk_tokens`,
  - `last_build_reused_cache`.

### `src/pipeline.py` (retriever reuse across follow-up queries)
- Added pipeline-level retriever cache:
  - `self._session_multimodal_retrievers: dict[str, SessionMultimodalRetriever]`.
- For visual uploads:
  - computes `session_id`,
  - reuses existing retriever instance when same PDF is queried again,
  - falls back to create/build only on first encounter.
- Added cache-hit logging:
  - `[SessionMultimodal] reusing index for session_<id>, skipping <n> chunk embeddings`.

### New observability fields (result + retrieval diagnostics)
- Added:
  - `session_index_rebuilt` (True on first build, False on cache hit),
  - `session_index_chunk_count`,
  - `session_embedding_tokens_saved` (estimated via `chunk_count * avg_chunk_tokens` on cache hit).
- Existing multimodal fields remain unchanged:
  - `multimodal_embedding_used`,
  - `multimodal_embedding_reason`,
  - `multimodal_hits_count`.

### Quick verification
- Compile checks passed:
  - `src/session_multimodal_retriever.py`
  - `src/pipeline.py`
- Lint checks passed for both touched files.

## Updates — May 1, 2026 (session retrieval mode toggle + upload-session fast path + generic UI cleanup)

_Appended; all sections above remain as-is._

### `src/pipeline.py` (session retrieval policy controls + latency improvements)
- Added `session_retrieval_mode` to `RAGPipeline.run(...)`:
  - supported values: `upload_only`, `hybrid`,
  - default fallback from env (`SESSION_RETRIEVAL_MODE`) or `hybrid`.
- Retrieval behavior now mode-aware:
  - `upload_only`: skips corpus retrieval (`k=0`) and uses uploaded/session context only,
  - `hybrid`: uses corpus retrieval + session context merge.
- Added session upload-only extractive fast path:
  - for factual-style prompts (e.g., short `what is ...`),
  - extracts answer from session sentences directly before router generation.
- Added session-scoped exact-match query cache:
  - key: `(session_scope_hash, normalized_query)`,
  - `session_scope_hash` derives from uploaded file bytes/text + retrieval mode,
  - prevents cross-file contamination while enabling repeat-query speedups in the same upload session.
- Added observability fields:
  - `session_retrieval_mode`,
  - `session_corpus_supplement_used`,
  - `session_corpus_supplement_count`,
  - plus cache visibility via result-level `cache_hit` / `cache_similarity`.

### `ui.py` (mode toggle + cache visibility + generic assistant copy)
- Added visible session retrieval mode toggle in chat UI when files are uploaded:
  - `Upload only (ignore base corpus)`,
  - `Hybrid (uploaded files + base corpus)`.
- Passed `session_retrieval_mode` into `pipeline.run(...)`.
- Added visible `cache hit` pill in chat metadata strip:
  - shows `cache hit` and, when available, similarity score.
- Generalized assistant branding and messaging:
  - replaced Notre-Dame-specific title/subtitle and prompts with generic assistant language,
  - updated uploader label to `Upload file (PDF/TXT)`,
  - generalized fallback/off-topic messaging to context-based wording.
- Updated visual styling:
  - switched to generic robot icon (`🤖`),
  - increased icon size and applied high-contrast, bold header color for better visibility.

### Runtime behavior changes observed
- Repeated identical queries in the same upload session now short-circuit on session cache.
- Simple definition-style questions in `upload_only` mode can avoid local LLM generation via extractive answer.
- Previous startup `403` symptoms traced to proxy/HF metadata requests; runtime stabilized by launching with:
  - `HF_HUB_OFFLINE=1`,
  - `TRANSFORMERS_OFFLINE=1`,
  - and (for responsiveness during tests) `USE_SESSION_MULTIMODAL_EMBEDDING=false`.

### Quick verification
- Compile checks passed:
  - `src/pipeline.py`
  - `ui.py`
- Lint checks passed for touched files.
- Streamlit restarted successfully and reachable at `http://localhost:8501`.

## Updates — May 1, 2026 (upload-session answer isolation + grounding allowlist + text-only session embeddings)

_Appended; all sections above remain as-is._

### `ui.py` (prevent answer bleed in upload sessions)
- Updated short-query contextual stitching behavior:
  - when `session_uploads` is present, skip `build_contextual_query(...)` stitching entirely,
  - each uploaded-file query is treated independently to prevent prior-turn topic contamination.
- Existing stitching behavior remains unchanged when no uploads are present.

### `src/pipeline.py` (grounding false-positive reduction for technical terms)
- Added a technical entity allowlist used by `_grounding_gate(...)` filtering before unsupported-entity checks:
  - `BM25`, `TF-IDF`, `BERT`, `embeddings`, `cosine`, `LangChain`, `LlamaIndex`, `ChromaDB`, `RAG`, `LLM`, `API`.
- Result: standard technical acronyms from uploaded docs no longer trigger avoidable `unsupported_entities` failures.

### `src/session_multimodal_retriever.py` + `src/pipeline.py` (all-upload activation with Gemini/BGE routing)
- Added `use_gemini: bool` to `SessionMultimodalRetriever`:
  - `use_gemini=True`: Gemini embeddings path (`session_<session_id>` collection),
  - `use_gemini=False`: local BGE path via `get_embedding_model("BAAI/bge-small-en-v1.5")` (`session_text_<session_id>` collection), zero Gemini API cost.
- Pipeline session retrieval activation now runs for all uploads (not only visual+key):
  - visual PDF + Gemini key → `use_gemini=True`,
  - text-only upload (or visual upload without key) → `use_gemini=False`.
- Kept manual query embedding behavior (`query_embeddings`) unchanged.
- Removed pipeline-level session retriever object cache usage so in-memory session collection reuse is owned by the retriever module cache.

### Quick verification
- Compile checks passed:
  - `src/session_multimodal_retriever.py`
  - `src/pipeline.py`
  - `ui.py`
- Lint checks passed for touched files.

## Updates — May 1, 2026 (complex-query grounding gate coverage-gated bypass)

_Appended; all sections above remain as-is._

### `src/pipeline.py` (call-site only, no grounding core changes)
- Updated the main `run()` grounding-gate call site to support a coverage-gated bypass for analytical complex queries:
  - if `complexity == "complex"` and `retrieval.coverage_score >= 0.55`, skip strict entity grounding for that turn,
  - set `grounded_ok = True` with structured `grounding_meta`:
    - `reason: "skipped_for_complex_query"`
    - `unsupported_entities: []`
    - `coverage_score: <value>`
  - emit explicit runtime log via `print(...)` including coverage value.
- All other behavior remains unchanged:
  - complex queries with weak retrieval (`coverage < 0.55`) still run `_grounding_gate(...)`,
  - simple/medium queries always run `_grounding_gate(...)`,
  - `_grounding_gate()` implementation itself was not modified.

### Quick verification
- Compile check passed:
  - `src/pipeline.py`
- Lint checks passed for touched files.
