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
        complexity = analysis["complexity_label"]
        print(
            f"Complexity: {complexity} "
            f"(confidence={analysis['confidence']}, source={analysis['source']}, llm_status={analysis['llm_status']})"
        )

        # Step 2 — Retrieve relevant chunks
        retrieval = self.retriever.retrieve(
            query=query,
            complexity=complexity,
            complexity_score=analysis.get("complexity_score"),
            analyzer_confidence=analysis.get("confidence"),
        )
        docs = retrieval["docs"]
        print(
            f"Retrieved {retrieval['k']} chunks "
            f"(k_base={retrieval['k_base']}, coverage={retrieval['coverage_score']}, retry={retrieval['retrieval_retry']}):"
        )
        for c in retrieval["chunks"]:
            print(f"  [{c['chunk_index']}] {c['source']} (score: {c['similarity_score']})")

        # Step 3 — Compress context
        compression = self.compressor.compress(query, docs, complexity)
        print(f"Tokens: {compression['original_token_count']} → {compression['compressed_token_count']} (ratio: {compression['compression_ratio']})")
        print(f"Sentences kept: {compression['selected_indices']} of {len(compression['all_sentences'])} total")

        # Step 4 — Route to appropriate model and generate answer
        router_result = self.router.route(query, complexity, compression)
        print(f"Model used: {router_result['model_used']} | Input tokens: {router_result['input_tokens']} | Output tokens: {router_result['output_tokens']} | Time: {router_result['time_taken_seconds']}s")
        print(f"Initial answer: {router_result['answer']}")

        # Step 5 — Confidence check and retry if needed
        confidence_result = check_confidence(
            query=query,
            compressed_context=compression["compressed_context"],
            router_output=router_result,
            analyzer_output=analysis,
        )
        breakdown = confidence_result["score_breakdown"]
        print(f"Score breakdown — heuristic: {breakdown['heuristic']} | tfidf: {breakdown['tfidf']} | embedding: {breakdown['embedding']} | semantic: {breakdown['semantic']}")
        print(f"Confidence: {confidence_result['confidence_score_final']} | Retried: {confidence_result['retried']}" + (f" | Reason: {confidence_result['retry_reason']}" if confidence_result["retried"] else ""))
        print(f"Final model: {confidence_result['model_used_final']}")
        print(f"Final answer: {confidence_result['final_answer']}")

        return {
            "query": query,
            "complexity": complexity,
            "complexity_source": analysis["source"],
            "complexity_score": analysis.get("complexity_score"),
            "complexity_confidence": analysis["confidence"],
            "complexity_reason_codes": analysis["reason_codes"],
            "query_analyzer_llm_status": analysis["llm_status"],
            # retrieval metadata
            "k": retrieval["k"],
            "retrieval_complexity_used": retrieval["complexity_used"],
            "retrieval_complexity_score_used": retrieval["complexity_score_used"],
            "retrieval_analyzer_confidence": retrieval["analyzer_confidence"],
            "k_base": retrieval["k_base"],
            "k_final": retrieval["k_final"],
            "coverage_score": retrieval["coverage_score"],
            "retrieval_retry": retrieval["retrieval_retry"],
            "chunks": retrieval["chunks"],
            # compression metadata
            "compressed_context": compression["compressed_context"],
            "all_sentences": compression["all_sentences"],
            "sentence_scores": compression["sentence_scores"],
            "selected_indices": compression["selected_indices"],
            "original_token_count": compression["original_token_count"],
            "compressed_token_count": compression["compressed_token_count"],
            "compression_ratio": compression["compression_ratio"],
            # router metadata
            "model_used_original": router_result["model_used"],
            "input_tokens": router_result["input_tokens"],
            "output_tokens": router_result["output_tokens"],
            "time_taken_seconds": router_result["time_taken_seconds"],
            # confidence metadata
            "score_breakdown": confidence_result["score_breakdown"],
            "confidence_score_original": confidence_result["confidence_score_original"],
            "confidence_score_final": confidence_result["confidence_score_final"],
            "retried": confidence_result["retried"],
            "retry_reason": confidence_result["retry_reason"],
            "model_used_final": confidence_result["model_used_final"],
            "final_answer": confidence_result["final_answer"],
        }


