import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from pipeline import RAGPipeline


def run_evaluation(queries_file: str = None, output_file: str = None):
    queries_file = queries_file or str(PROJECT_ROOT / "data" / "test_queries.csv")
    output_file = output_file or str(PROJECT_ROOT / "evaluation_results.csv")

    pipeline = RAGPipeline()

    rows = []
    total = 0
    retrieval_correct = 0

    with open(queries_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row["query_id"]
            query = row["query"]
            complexity = row["complexity"]
            ground_truth = row["ground_truth"]
            ground_truth_source = row.get("ground_truth_source", "").strip()

            result = pipeline.run(query)

            # Retrieval accuracy — only checkable when ground_truth_source is known
            retrieved_sources = [c["source"] for c in result["chunks"]]
            if ground_truth_source:
                hit = any(ground_truth_source in src for src in retrieved_sources)
                retrieval_hit = "yes" if hit else "no"
                total += 1
                if hit:
                    retrieval_correct += 1
            else:
                retrieval_hit = "n/a"

            rows.append({
                "query_id": query_id,
                "query": query,
                "complexity": complexity,
                "ground_truth": ground_truth,
                "ground_truth_source": ground_truth_source,
                "retrieved_sources": " | ".join(retrieved_sources),
                "retrieval_hit": retrieval_hit,
                "k": result["k"],
                "original_tokens": result["original_token_count"],
                "compressed_tokens": result["compressed_token_count"],
                "compression_ratio": result["compression_ratio"],
                "model_original": result["model_used_original"],
                "model_final": result["model_used_final"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "time_seconds": result["time_taken_seconds"],
                "confidence_original": result["confidence_score_original"],
                "confidence_final": result["confidence_score_final"],
                "retried": result["retried"],
                "retry_reason": result.get("retry_reason") or "",
                "heuristic_score": result["score_breakdown"]["heuristic"],
                "tfidf_score": result["score_breakdown"]["tfidf"],
                "embedding_score": result["score_breakdown"]["embedding"],
                "final_answer": result["final_answer"],
            })

    # Write results CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries run       : {len(rows)}")
    if total > 0:
        retrieval_accuracy = round(retrieval_correct / total * 100, 1)
        print(f"Retrieval accuracy      : {retrieval_correct}/{total} ({retrieval_accuracy}%)")
    retried_count = sum(1 for r in rows if r["retried"] == True or r["retried"] == "True")
    print(f"Queries retried         : {retried_count}/{len(rows)}")
    avg_compression = round(sum(float(r["compression_ratio"]) for r in rows) / len(rows), 4)
    print(f"Avg compression ratio   : {avg_compression}")
    avg_confidence = round(sum(float(r["confidence_final"]) for r in rows) / len(rows), 4)
    print(f"Avg confidence score    : {avg_confidence}")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    run_evaluation()
