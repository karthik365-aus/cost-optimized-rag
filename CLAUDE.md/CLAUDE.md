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
