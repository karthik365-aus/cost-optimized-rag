# Team Guide — Cost Optimized RAG

## Project Overview

A cost-optimized Retrieval Augmented Generation (RAG) pipeline that classifies query complexity and uses compression to reduce LLM token usage.

---

## Pipeline Flow

```
User Query
    ↓
[DONE] Query analyzer       → complexity label + score (+ optional FLAN)
    ↓
[DONE] Adaptive retrieval   → Chroma + dynamic k / coverage retry
    ↓
[DONE] Context compression   → hybrid TF-IDF / embeddings, adaptive budget
    ↓
[DONE] Model router          → simple/medium → local OpenAI-compatible API; complex → OpenAI
    ↓
[DONE] Confidence checker    → answer vs compressed context; OpenAI tier retry (see MODEL_TIERS note in README)
    ↓
Final Answer
```

---

## What's on GitHub — File Reference

```
cost-optimized-rag/
│
├── src/
│   ├── adaptive_retriever.py         ✅ Chroma + BGE embeddings
│   ├── context_compression.py        ✅ sentence selection / hybrid scoring
│   ├── model_router.py               ✅ local OpenAI-compatible + OpenAI complex
│   ├── confidence_checker.py         ✅ grounding score; tier retry for OpenAI model names
│   ├── query_analyzer.py             ✅ complexity heuristics + optional FLAN
│   ├── pipeline.py                   ✅ end-to-end orchestration
│   ├── preflight.py                  ✅ CSV / Chroma / local models checks for evaluate
│   └── seed_manager.py               ✅ reproducible seeds
│
├── data/
│   ├── documents/                    ✅ source `.txt` files for ingestion
│   └── test_queries.csv              ✅ benchmark queries + ground truth
│
├── evaluate.py                       ✅ full pipeline benchmark → CSV + JSON (+ CLI)
├── scripts/run_eval.sh               ✅ run evaluate.py from repo root
├── chroma_db/                        (generated) vector index under project root
├── requirements.txt
└── .env                              ⚠️  NOT on GitHub — create from `.env.example`
```

---

## Getting Started (do this once)

### 1. Clone the repo
```bash
git clone https://github.com/karthik365-aus/cost-optimized-rag.git
cd cost-optimized-rag
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create your .env file
Create `.env` in the project root (not on GitHub). Start from `.env.example`: OpenAI key for **complex** queries, `LOCAL_OPENAI_*` and exact local model IDs for **simple/medium**, optional compression and `EVAL_GT_EMBEDDING_MODEL` vars.

### 5. Run evaluation (full pipeline + benchmark metrics)
From the project root, with LM Studio (or your local server) running if you use local models:

```bash
python evaluate.py --help
python evaluate.py
# or: ./scripts/run_eval.sh --seed 7 -v
```

Preflight validates the CSV, Chroma or document paths, and (by default) local `/v1/models`. See root **README.md** for flags (`--no-preflight`, `--skip-local-model-check`, outputs, logging).

---

## Daily Git Workflow

### Before starting work — always pull latest
```bash
git pull
```

### After finishing your work — push to GitHub
```bash
git add src/model_router.py          # add only your file
git commit -m "Add model router logic"
git push
```

---

## For Model Router Team

**File to work in:** `src/model_router.py`

**Your input** comes from `ContextCompressor.compress()`:
```python
{
    "compressed_context": "...",       # the text to send to the LLM
    "compression_ratio": 0.97,
    "original_token_count": 312,
    "compressed_token_count": 10,
}
```
Plus: `query` (string) and `complexity` (`"simple"` / `"medium"` / `"complex"`)

**Your job:** Map complexity to a model. **Current implementation (see `.env.example`):**  
`simple` / `medium` → **local** OpenAI-compatible server (e.g. LM Studio) using `LOCAL_SIMPLE_MODEL` / `LOCAL_MEDIUM_MODEL` from `/v1/models`; `complex` → **OpenAI** via `OPENAI_COMPLEX_MODEL` and `OPENAI_API_KEY`.

**Your output** is a dict like (see `src/model_router.py` for full fields):
```python
{
    "answer": "The answer text...",
    "model_used": "<id from env>",
    "model_source": "local_openai_compatible" or "openai",
    "complexity": "simple"
}
```

**Reference:** `src/pipeline.py` calls the router right after `ContextCompressor.compress()`. For an end-to-end test, run `python evaluate.py` from the project root.

---

## For Confidence Checker Team

**File to work in:** `src/confidence_checker.py`

**Your input** comes from Model Router output:
```python
{
    "answer": "The answer text...",
    "model_used": "gpt-3.5-turbo",
    "complexity": "simple"
}
```
Plus: the original `query` and `compressed_context`

**Your job:** Score the answer **against the compressed context** (grounding). If confidence is low, `MODEL_TIERS` can trigger a one-step retry to a **stronger OpenAI** model when `model_used` matches a tier key. **Local LM Studio model ids do not match those keys, so retry usually does not run for local calls** (scores are still returned).

**Your output** should be:
```python
{
    "final_answer": "The answer text...",
    "confidence_score": 0.87,
    "model_used": "gpt-3.5-turbo",
    "retried": False
}
```

---

## Running Tests

| Command | What it does |
|---|---|
| `python evaluate.py` | Runs every row in `data/test_queries.csv` through `RAGPipeline`; writes `evaluation_results.{csv,json}` (override with `--out-csv` / `--out-json`) |
| `python evaluate.py --queries path/to.csv --seed 7` | Custom benchmark file and seed |
| `./scripts/run_eval.sh …` | Same as `python evaluate.py …` from repo root |

---

## Rules

- Never commit `.env` — it contains your API key
- Always `git pull` before starting work
- Only add your own files when doing `git add` — don't use `git add .`
- If you get a merge conflict, ask the team before force pushing

---

## Recent updates (Apr 24, 2026 — appended)

- **End-to-end run:** Use **`python evaluate.py`** from the project root (or **`scripts/run_eval.sh`**). **`evaluate.py`** loads **`.env` before preflight** so LM Studio model IDs are picked up; it also **creates output directories** for custom `--out-csv` / `--out-json` paths.
- **Compression ↔ retrieval:** **`coverage_score`** from retrieval is fed into **`context_compression`** to adjust how many sentences are kept when keyword overlap with retrieved text is weak.
- **Retrieval JSON fields:** Prefer **`chroma_distance`** for chunk ranking diagnostics (**lower = closer**); **`similarity_score`** is a legacy alias for the same raw value.
- **Confidence checker:** Scores answer vs **compressed context**, not CSV ground truth; **`MODEL_TIERS`** retry targets **OpenAI-style** model names — **local LM Studio ids usually won’t trigger** a tier retry (documented in code + README).
- **Query analyzer:** FLAN override threshold follows **`QUERY_ANALYZER_HIGH_CONF_THRESHOLD`** (env), not a hardcoded mid value — check **`.env.example`** / README if behavior feels “stricter” on FLAN adoption.
- **Model router:** No per-complexity temperature or prompt-template fork in code right now (a Phase-2 variant was **reverted**); single prompt + default temperature **0.0** unless you change **`ModelRouter`** construction in code.
- **Git hygiene:** **`.gitignore`** now ignores **`results/`**, **`.hf_cache/`**, **`venv/`**, Office temp **`~$*`** and **`*.docx`** — still review `git status` before push.
