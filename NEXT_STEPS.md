# NEXT_STEPS.md

## Current Project State (Apr 25, 2026)

- Semantic chunking is integrated in `src/adaptive_retriever.py` with fallback:
  - `USE_SEMANTIC_CHUNKER=true|false`
  - `SEMANTIC_BREAKPOINT_TYPE` (default `percentile`)
  - `SEMANTIC_BREAKPOINT_AMOUNT` (default `85`)
- Query-time HyDE is integrated in `src/pipeline.py` (flag-gated):
  - `USE_HYDE=true|false` (default `false`)
  - `HYDE_MIN_COVERAGE_GAIN` (default `0.03`)
  - `HYDE_MAX_CHARS` (default `700`)
- Retrieval diagnostics are surfaced in `ui.py` chat diagnostics panel.
- Corpus curation fix added to `data/documents/doc_18.txt`:
  - canonical founder sentence for Notre Dame is now present.
- Chroma index was rebuilt after corpus/chunking changes.
- Latest pushed commit: `264c52c` on `main`.

## What Was Validated

- Founder query now returns grounded factual answer after corpus update:
  - `Who founded Notre Dame?` -> Father Edward Sorin sentence.
- Factual extraction-first branch works for factual prompts.
- Grounding safeguards still active for unsupported content.
- UI runs with diagnostics and HyDE visibility.

## Resume Commands (Next Session)

### 1) Environment

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Start UI (recommended)

Without HyDE:

```bash
streamlit run ui.py --server.port 8501
```

With HyDE:

```bash
USE_HYDE=true streamlit run ui.py --server.port 8501
```

### 3) Quick smoke test

```bash
python - <<'PY'
from src.pipeline import RAGPipeline
p = RAGPipeline()
for q in [
    "Who founded Notre Dame?",
    "What is the Hesburgh Library?",
    "How many undergraduate colleges are at Notre Dame?",
]:
    r = p.run(q)
    print("\\nQ:", q)
    print("A:", r.get("final_answer"))
    print("model:", r.get("model_used_final"))
    print("confidence:", r.get("confidence_score_final"))
PY
```

## Open/Pending Items

- Decide whether to keep HyDE enabled by default after A/B evaluation.
- Consider removing now-unused `force_k` plumbing in retriever (cleanup only).
- Optional: standardize old legacy files (`streamlit_app.py` vs `ui.py`) to reduce confusion.
- Optional: migrate deprecated LangChain imports (`HuggingFaceEmbeddings`, `Chroma`) when convenient.

## Notes for Future Sessions

- If retrieval behavior changes unexpectedly, rebuild index by deleting `chroma_db/` and reinitializing `RAGPipeline()`.
- Keep appending major changes to:
  - `CLAUDE.md/CLAUDE.md` (full technical log)
  - `README.md` (user-facing update summary)
