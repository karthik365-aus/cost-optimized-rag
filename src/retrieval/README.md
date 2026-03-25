# Adaptive Retrieval Component

## Overview
Dynamically retrieves relevant documents based on query complexity.

## Requirements
- Python 3.10+
- OpenAI API key
- Libraries: langchain, chromadb, openai

## Setup

1. Install dependencies:
```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters chromadb python-dotenv
```

2. Create `.env` file in project root:
```
OPENAI_API_KEY=your-key-here
```

3. Ensure data folder has documents:
```
data/documents/doc_00.txt ... doc_19.txt
```

## Usage

### Run standalone test:
```bash
python src/retrieval/adaptive_retriever.py
```

### Use in your code:
```python
from src.retrieval.adaptive_retriever import AdaptiveRetriever

retriever = AdaptiveRetriever()
docs = retriever.retrieve("What are office hours?", "simple")
```

## Expected Output
```
Loading documents...
✅ Loaded 20 documents
✅ Created 30 chunks
Creating vector database...
✅ Vector database created

SIMPLE: Retrieved 3 documents
MEDIUM: Retrieved 5 documents
COMPLEX: Retrieved 10 documents
```

## Component Behavior

**Complexity-based retrieval:**
- `simple` → 3 documents
- `medium` → 5 documents
- `complex` → 10 documents

**Vector search:** Uses OpenAI embeddings + Chroma for semantic similarity.

## Testing

Run test file:
```bash
python test_chroma.py
```

## Troubleshooting

**Error: No module named 'langchain'**
→ Run: `pip install langchain-community`

**Error: OpenAI API key not found**
→ Check `.env` file exists with correct key

**Error: No documents found**
→ Verify `data/documents/` has .txt files

## Notes
- First run creates `chroma_db/` folder (takes ~30 seconds)
- Subsequent runs are faster (uses cached embeddings)
- Cost: ~$0.01 for initial embedding creation

## Contact
Questions? Ask Karthik in Teams.