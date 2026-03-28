# Team Guide — Cost Optimized RAG

## Project Overview

A cost-optimized Retrieval Augmented Generation (RAG) pipeline that classifies query complexity and uses compression to reduce LLM token usage.

---

## Pipeline Flow

```
User Query
    ↓
[DONE] Adaptive Retrieval   → retrieves 3/5/10 relevant chunks from ChromaDB
    ↓
[DONE] Context Compression  → compresses chunks to top 2/4/6 sentences
    ↓
[TODO] Model Router         → routes to cheap/mid/powerful LLM based on complexity
    ↓
[TODO] Confidence Checker   → validates response quality, retries if needed
    ↓
Final Answer
```

---

## What's on GitHub — File Reference

```
cost-optimized-rag/
│
├── src/
│   ├── retrieval/
│   │   └── adaptive_retriever.py     ✅ DONE (Karthik) — vector search, returns chunks
│   ├── context_compression.py        ✅ DONE (Anh)     — compresses chunks to key sentences
│   ├── model_router.py               🔲 TODO (assign)  — pick LLM based on complexity
│   ├── confidence_checker.py         🔲 TODO (assign)  — validate and retry LLM response
│   ├── query_analyzer.py             🔲 TODO           — classify query complexity
│   └── adaptive_retrieval.py         🔲 TODO           — orchestrates full pipeline
│
├── data/
│   ├── documents/                    ✅ source text files (Notre Dame dataset)
│   └── test_queries.csv              ✅ 50 test queries (simple/medium/complex)
│
├── test_pipeline.py                  ✅ run this to test retrieval + compression together
├── run_all_queries.py                ✅ run this to evaluate all 50 queries, saves results to CSV
├── requirements.txt                  ✅ all dependencies
└── .env                              ⚠️  NOT on GitHub — you must create this yourself (see below)
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
Create a file called `.env` in the project root (this is NOT on GitHub for security):
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Test the pipeline works
```bash
python test_pipeline.py
```
You should see retrieved chunks and a compressed context printed in the terminal.

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

**Your job:** Pick the right LLM model based on complexity and return the answer.

```python
# Model mapping (suggested)
simple   → gpt-3.5-turbo    (cheap, fast)
medium   → gpt-4o-mini      (balanced)
complex  → gpt-4o           (powerful)
```

**Your output** should be a dict like:
```python
{
    "answer": "The answer text...",
    "model_used": "gpt-3.5-turbo",
    "complexity": "simple"
}
```

**Reference:** Look at `test_pipeline.py` to see how retrieval and compression connect — your module plugs in right after compression.

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

**Your job:** Score the answer quality. If confidence is low, retry with a stronger model.

```python
# Suggested logic
if confidence_score < threshold:
    retry with next tier model (e.g. gpt-3.5 → gpt-4o-mini)
else:
    return final answer
```

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
| `python test_pipeline.py` | Tests 1 query through retrieval + compression |
| `python run_all_queries.py` | Tests all 50 queries, saves to `compression_results.csv` |

---

## Rules

- Never commit `.env` — it contains your API key
- Always `git pull` before starting work
- Only add your own files when doing `git add` — don't use `git add .`
- If you get a merge conflict, ask the team before force pushing
