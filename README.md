# Cost-optimized RAG

RAG pipeline for the Notre Dame corpus with adaptive complexity routing, context compression, semantic cache, and hybrid local + OpenAI calls.

Primary entry points:
- `evaluate.py` for benchmark runs (`CSV` + `JSON` output)
- `src/pipeline.py` for one-query end-to-end runs

## Quick Start

```bash
git clone <repository-url>
cd cost-optimized-rag
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Set at least these in `.env`:
- `OPENAI_API_KEY=...`
- `LOCAL_OPENAI_BASE_URL=http://127.0.0.1:1234/v1`
- `LOCAL_SIMPLE_MODEL=<id-from-/v1/models>`
- `LOCAL_MEDIUM_MODEL=<id-from-/v1/models>`

Verify local model IDs:

```bash
curl -s http://127.0.0.1:1234/v1/models | python3 -m json.tool
```

## For Teammates: 60-second Run Checklist

1. Clone/download the full repo (not a single file).
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and set:
   - `OPENAI_API_KEY`
   - `LOCAL_OPENAI_BASE_URL` (with `/v1`)
   - `LOCAL_SIMPLE_MODEL`
   - `LOCAL_MEDIUM_MODEL`
4. Start LM Studio local server and load local model(s).
5. Run smoke test:
   - `python evaluate.py --queries data/smoke_queries.csv --out-csv results/smoke.csv --out-json results/smoke.json`
6. If smoke passes, run full benchmark:
   - `python evaluate.py --out-csv results/eval.csv --out-json results/eval.json`

## What This Pipeline Does

- `src/query_analyzer.py`: predicts `complexity_label`, `complexity_score`, `confidence`
- `src/adaptive_retriever.py`: Chroma retrieval with dynamic `k` and coverage-aware retry
- `src/context_compression.py`: TF-IDF + embedding sentence selection with adaptive budget
- `src/model_router.py`: `simple/medium` to local OpenAI-compatible endpoint, `complex` to OpenAI
- `src/confidence_checker.py`: grounding score vs compressed context, optional stronger-model retry (OpenAI-style model keys only)
- `src/semantic_cache.py`: near-duplicate query cache (same corpus hash + high semantic similarity)

## Component Owner Map

| Component | File | Owner |
|---|---|---|
| Query Analyzer | `src/query_analyzer.py` | Evelyn |
| Adaptive Retriever | `src/adaptive_retriever.py` | Karthik |
| Context Compressor | `src/context_compression.py` | Anh |
| Model Router | `src/model_router.py` | Gowri |
| Confidence Checker | `src/confidence_checker.py` | Burcu |
| Pipeline Orchestration | `src/pipeline.py` | Karthik |
| Evaluation Driver | `evaluate.py` | Karthik |

## Run Commands

Smoke run (fast, 10 queries):

```bash
python evaluate.py \
  --queries data/smoke_queries.csv \
  --out-csv results/smoke.csv \
  --out-json results/smoke.json
```

Full benchmark:

```bash
python evaluate.py \
  --out-csv results/eval.csv \
  --out-json results/eval.json
```

Faster iteration (TF-IDF-only compression):

```bash
COMPRESSION_USE_EMBEDDINGS=false python evaluate.py --queries data/smoke_queries.csv
```

Single-query run:

```python
from src.pipeline import RAGPipeline
pipeline = RAGPipeline()
result = pipeline.run("What is the Hesburgh Library?")
print(result["final_answer"])
```

## Expected Output

From `evaluate.py` you should get:
- output files at the `--out-csv` and `--out-json` paths
- console summary with hit rate, failure breakdown, and latency stats

Useful columns to inspect first:
- `answer_correct`
- `failure_type` (`TYPE_A`, `TYPE_B`, `TYPE_C`)
- `cache_hit`
- `confidence_score_final`
- `compression_ratio`
- `retrieval_avg_chunk_distance` (lower is better)

## Important Runtime Behavior

- Preflight is on by default.
- If CSV contains any `complex` rows and `OPENAI_API_KEY` is missing, preflight fails fast.
- Retriever score is stored as `chroma_distance` (distance, lower is better); legacy `similarity_score` may still appear for compatibility.
- Confidence retry uses `MODEL_TIERS` and only applies when model names match those keys (local LM Studio IDs usually do not match).
- Pipeline now applies **general typo-tolerant query normalization** (corpus-driven fuzzy correction), then may run a stronger fallback retrieval pass when answer quality is low.

## Troubleshooting

- `Local server does not list configured model(s)`:
  - Ensure `LOCAL_SIMPLE_MODEL` and `LOCAL_MEDIUM_MODEL` exactly match `/v1/models` IDs.
- `OPENAI_API_KEY is unset`:
  - Add key, run with `--require-openai-key`, or use a non-complex query set.
- `HF` permission / download issues:
  - Set `HF_HOME` to a writable path (e.g. `./.hf_cache`).
- Many `TYPE_B` failures:
  - Increase compression budget or reduce redundancy strictness.
- Query has misspellings or noisy wording:
  - The pipeline attempts auto-correction and fallback retrieval automatically; if results are still weak, rephrase with one concrete entity phrase (e.g., full proper noun) to improve retrieval coverage.

## Repository Layout

```text
cost-optimized-rag/
├── README.md
├── CLAUDE.md/CLAUDE.md
├── TEAM_GUIDE.md
├── evaluate.py
├── .env.example
├── scripts/run_eval.sh
├── data/{documents,test_queries.csv,smoke_queries.csv}
├── chroma_db/
└── src/
    ├── pipeline.py
    ├── query_analyzer.py
    ├── adaptive_retriever.py
    ├── context_compression.py
    ├── model_router.py
    ├── confidence_checker.py
    ├── semantic_cache.py
    ├── shared_embeddings.py
    ├── preflight.py
    └── seed_manager.py
```

## Further Documentation

- `CLAUDE.md/CLAUDE.md` for implementation history and design decisions
- `TEAM_GUIDE.md` for teammate workflow and ownership notes

## Recent Updates (Apr 25, 2026)

- **Semantic chunking option added in retriever**
  - `src/adaptive_retriever.py` now supports `SemanticChunker` (topic-aware splits) with fallback to `RecursiveCharacterTextSplitter`.
  - New env flags:
    - `USE_SEMANTIC_CHUNKER` (default `true`)
    - `SEMANTIC_BREAKPOINT_TYPE` (default `percentile`)
    - `SEMANTIC_BREAKPOINT_AMOUNT` (default `85`)
  - `requirements.txt` now includes `langchain-experimental`.
  - Chroma rebuild is auto-triggered on chunking config/version change.

- **Optional HyDE retrieval in pipeline**
  - `src/pipeline.py` now supports query-time HyDE behind `USE_HYDE` (default `false`).
  - Flow: generate one hypothetical retrieval text -> run alternate retrieval -> adopt only if coverage improves.
  - Tunables:
    - `HYDE_MIN_COVERAGE_GAIN` (default `0.03`)
    - `HYDE_MAX_CHARS` (default `700`)
  - Result payload includes: `hyde_attempted`, `hyde_used`, `hyde_coverage_score`, `hyde_query_preview`, `retrieval_query_used`.

- **UI diagnostics expanded**
  - `ui.py` retrieval diagnostics now shows HyDE attempted/used and alternate coverage score.

- **Founder-answer corpus curation**
  - Added canonical fact sentence to `data/documents/doc_18.txt`:
    - “Father Edward Sorin of the Congregation of Holy Cross founded the University of Notre Dame on November 26, 1842.”
  - Rebuilt `chroma_db` after this update.
  - Validation now returns the canonical grounded founder answer for `Who founded Notre Dame?`.
