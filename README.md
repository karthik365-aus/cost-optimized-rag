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

## Troubleshooting

- `Local server does not list configured model(s)`:
  - Ensure `LOCAL_SIMPLE_MODEL` and `LOCAL_MEDIUM_MODEL` exactly match `/v1/models` IDs.
- `OPENAI_API_KEY is unset`:
  - Add key, run with `--require-openai-key`, or use a non-complex query set.
- `HF` permission / download issues:
  - Set `HF_HOME` to a writable path (e.g. `./.hf_cache`).
- Many `TYPE_B` failures:
  - Increase compression budget or reduce redundancy strictness.

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
