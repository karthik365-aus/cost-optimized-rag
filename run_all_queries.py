import sys
from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.adaptive_retriever import AdaptiveRetriever
from src.context_compression import ContextCompressor
from src.model_router import ModelRouter


def main():
    documents_path = str(PROJECT_ROOT / "data" / "documents")
    queries_file = PROJECT_ROOT / "data" / "test_queries.csv"

    retriever = AdaptiveRetriever(documents_path=documents_path)
    compressor = ContextCompressor()
    router = ModelRouter()

    results = []

    with open(queries_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            query = row["query"]
            complexity = row["complexity"]

            print("\n" + "=" * 80)
            print("Query:", query)
            print("Complexity:", complexity)

            docs = retriever.retrieve(query, complexity)
            result = compressor.compress(query, docs, complexity)
            router_result = router.route(query, complexity, result)

            print("Original tokens:", result["original_token_count"])
            print("Compressed tokens:", result["compressed_token_count"])
            print("Compression ratio:", result["compression_ratio"])
            print("Model used:", router_result["model_used"])

            results.append([
                query,
                complexity,
                result["original_token_count"],
                result["compressed_token_count"],
                result["compression_ratio"],
                router_result["model_used"],
                router_result["answer"],
            ])

    output_file = PROJECT_ROOT / "compression_results.csv"
    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "query",
            "complexity",
            "original_tokens",
            "compressed_tokens",
            "compression_ratio",
            "model_used",
            "answer",
        ])
        writer.writerows(results)

    print("\nResults saved to compression_results.csv")


if __name__ == "__main__":
    main()
