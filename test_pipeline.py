import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.adaptive_retriever import AdaptiveRetriever
from src.context_compression import ContextCompressor
from src.model_router import ModelRouter


def main():
    query = "What are the office hours?"
    complexity = "simple"

    documents_path = str(PROJECT_ROOT / "data" / "documents")

    retriever = AdaptiveRetriever(documents_path=documents_path)
    docs = retriever.retrieve(query, complexity)

    print("\n=== RETRIEVED CHUNKS ===")
    for i, doc in enumerate(docs, start=1):
        print(f"\nChunk {i}:")
        print(repr(doc.page_content))

    compressor = ContextCompressor()
    result = compressor.compress(query, docs, complexity)

    print("\n=== COMPRESSED CONTEXT ===")
    print(result["compressed_context"])

    print("\n=== METRICS ===")
    print("Original tokens:", result["original_token_count"])
    print("Compressed tokens:", result["compressed_token_count"])
    print("Compression ratio:", result["compression_ratio"])

    router = ModelRouter()
    router_result = router.route(query, complexity, result)

    print("\n=== MODEL ROUTER OUTPUT ===")
    print("Model used:", router_result["model_used"])
    print("Answer:", router_result["answer"])


if __name__ == "__main__":
    main()
