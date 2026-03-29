import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.query_analyzer import QueryAnalyzer
from src.adaptive_retriever import AdaptiveRetriever
from src.context_compression import ContextCompressor
from src.model_router import ModelRouter
from src.confidence_checker import check_confidence


class RAGPipeline:
    def __init__(self, documents_path=None):
        documents_path = documents_path or str(PROJECT_ROOT / "data" / "documents")
        self.analyzer = QueryAnalyzer()
        self.retriever = AdaptiveRetriever(documents_path=documents_path)
        self.compressor = ContextCompressor()
        self.router = ModelRouter()

    def run(self, query: str) -> dict:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        # Step 1 — Classify query complexity
        analysis = self.analyzer.analyze(query)
        complexity = analysis["complexity"]
        print(f"Complexity: {complexity}")

        # Step 2 — Retrieve relevant chunks
        docs = self.retriever.retrieve(query, complexity)

        # Step 3 — Compress context
        compression = self.compressor.compress(query, docs, complexity)
        print(f"Tokens: {compression['original_token_count']} → {compression['compressed_token_count']} (ratio: {compression['compression_ratio']})")

        # Step 4 — Route to appropriate model and generate answer
        router_result = self.router.route(query, complexity, compression)
        print(f"Model used: {router_result['model_used']}")
        print(f"Initial answer: {router_result['answer']}")

        # Step 5 — Confidence check and retry if needed
        confidence_result = check_confidence(
            query=query,
            compressed_context=compression["compressed_context"],
            router_output=router_result,
        )
        print(f"Confidence score: {confidence_result['confidence_score_final']}")
        print(f"Retried: {confidence_result['retried']} | Final model: {confidence_result['model_used_final']}")
        print(f"Final answer: {confidence_result['final_answer']}")

        return {
            "query": query,
            "complexity": complexity,
            "compressed_context": compression["compressed_context"],
            "original_token_count": compression["original_token_count"],
            "compressed_token_count": compression["compressed_token_count"],
            "compression_ratio": compression["compression_ratio"],
            "model_used_original": router_result["model_used"],
            "model_used_final": confidence_result["model_used_final"],
            "confidence_score": confidence_result["confidence_score_final"],
            "retried": confidence_result["retried"],
            "final_answer": confidence_result["final_answer"],
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()

    test_queries = [
        "What is the oldest structure at Notre Dame?",
        "How does student funding affect editorial independence?",
        "Analyze the trade-offs between faculty oversight and editorial independence in student publications",
    ]

    for query in test_queries:
        pipeline.run(query)
