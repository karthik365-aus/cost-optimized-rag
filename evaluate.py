import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from pipeline import RAGPipeline


# -----------------------------
# ANSWER CORRECTNESS
# -----------------------------
def is_answer_correct(ground_truth: str, llm_answer: str) -> bool:
    """Case-insensitive substring check — ground truth must appear in the LLM answer."""
    if not ground_truth or ground_truth.lower() in ("manual evaluation needed", "requires multi-step analysis"):
        return None  # can't auto-evaluate
    return ground_truth.lower() in llm_answer.lower()


# -----------------------------
# FIND WHICH SENTENCE CONTAINS THE ANSWER
# -----------------------------
def find_answer_sentence_index(sentences: list, ground_truth: str) -> int:
    """Return index of the first sentence containing the ground truth answer, or -1."""
    if not ground_truth:
        return -1
    for i, sentence in enumerate(sentences):
        if ground_truth.lower() in sentence.lower():
            return i
    return -1


# -----------------------------
# FAILURE CATEGORIZATION
# -----------------------------
def categorize_failure(
    ground_truth_source: str,
    retrieved_sources: list,
    all_sentences: list,
    selected_indices: list,
    ground_truth: str,
    llm_answer: str,
) -> str:
    correct = is_answer_correct(ground_truth, llm_answer)

    # Can't categorize if no ground truth source or manual evaluation needed
    if not ground_truth_source or correct is None:
        return "N/A"

    if correct:
        return "CORRECT"

    # Type A: correct chunk was never retrieved
    if not any(ground_truth_source in src for src in retrieved_sources):
        return "TYPE_A"

    # Type B: chunk retrieved but answer sentence was compressed away
    answer_sentence_idx = find_answer_sentence_index(all_sentences, ground_truth)
    if answer_sentence_idx != -1 and answer_sentence_idx not in selected_indices:
        return "TYPE_B"

    # Type C: context had the answer, LLM got it wrong
    return "TYPE_C"


# -----------------------------
# MAIN EVALUATION
# -----------------------------
def run_evaluation(queries_file: str = None, output_csv: str = None, output_json: str = None):
    queries_file = queries_file or str(PROJECT_ROOT / "data" / "test_queries.csv")
    output_csv = output_csv or str(PROJECT_ROOT / "evaluation_results.csv")
    output_json = output_json or str(PROJECT_ROOT / "evaluation_results.json")

    pipeline = RAGPipeline()

    rows = []
    json_records = []

    with open(queries_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row["query_id"]
            query = row["query"]
            complexity = row["complexity"]
            ground_truth = row["ground_truth"].strip()
            ground_truth_source = row.get("ground_truth_source", "").strip()

            result = pipeline.run(query)

            retrieved_sources = [c["source"] for c in result["chunks"]]
            all_sentences = result["all_sentences"]
            selected_indices = result["selected_indices"]
            llm_answer = result["final_answer"]

            correct = is_answer_correct(ground_truth, llm_answer)
            answer_sentence_idx = find_answer_sentence_index(all_sentences, ground_truth)
            failure_type = categorize_failure(
                ground_truth_source,
                retrieved_sources,
                all_sentences,
                selected_indices,
                ground_truth,
                llm_answer,
            )
            retrieval_hit = (
                "yes" if any(ground_truth_source in src for src in retrieved_sources)
                else "no" if ground_truth_source else "n/a"
            )

            csv_row = {
                "query_id": query_id,
                "query": query,
                "complexity": complexity,
                "ground_truth": ground_truth,
                "ground_truth_source": ground_truth_source,
                "answer_correct": "" if correct is None else ("yes" if correct else "no"),
                "failure_type": failure_type,
                "retrieved_sources": " | ".join(retrieved_sources),
                "retrieval_hit": retrieval_hit,
                "answer_sentence_idx": answer_sentence_idx,
                "answer_sentence_selected": (
                    "yes" if answer_sentence_idx in selected_indices
                    else "no" if answer_sentence_idx != -1
                    else "n/a"
                ),
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
                "final_answer": llm_answer,
            }
            rows.append(csv_row)

            # Full JSON record with all sentences and scores for deep-dive analysis
            json_records.append({
                **csv_row,
                "all_sentences": all_sentences,
                "sentence_scores": result["sentence_scores"],
                "selected_indices": selected_indices,
                "chunks": result["chunks"],
            })

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Write JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2)

    # --- Summary ---
    evaluatable = [r for r in rows if r["answer_correct"] in ("yes", "no")]
    correct_count = sum(1 for r in evaluatable if r["answer_correct"] == "yes")
    wrong_count = sum(1 for r in evaluatable if r["answer_correct"] == "no")

    type_a = sum(1 for r in rows if r["failure_type"] == "TYPE_A")
    type_b = sum(1 for r in rows if r["failure_type"] == "TYPE_B")
    type_c = sum(1 for r in rows if r["failure_type"] == "TYPE_C")

    retrieval_evaluatable = [r for r in rows if r["retrieval_hit"] in ("yes", "no")]
    retrieval_correct = sum(1 for r in retrieval_evaluatable if r["retrieval_hit"] == "yes")

    retried_count = sum(1 for r in rows if str(r["retried"]).lower() == "true")
    avg_compression = round(sum(float(r["compression_ratio"]) for r in rows) / len(rows), 4)
    avg_confidence = round(sum(float(r["confidence_final"]) for r in rows) / len(rows), 4)

    print(f"\n{'='*60}")
    print(f"FAILURE ANALYSIS REPORT")
    print(f"{'='*60}")
    print(f"Total queries run       : {len(rows)}")
    print(f"Auto-evaluatable        : {len(evaluatable)} (simple queries with ground truth)")
    if evaluatable:
        accuracy = round(correct_count / len(evaluatable) * 100, 1)
        print(f"Correct                 : {correct_count}/{len(evaluatable)} ({accuracy}%)")
        print(f"Wrong                   : {wrong_count}/{len(evaluatable)}")
    if retrieval_evaluatable:
        ret_acc = round(retrieval_correct / len(retrieval_evaluatable) * 100, 1)
        print(f"Retrieval accuracy      : {retrieval_correct}/{len(retrieval_evaluatable)} ({ret_acc}%)")

    if wrong_count > 0:
        print(f"\nBREAKDOWN OF {wrong_count} FAILURES:")
        print(f"  Type A (Retrieval)    : {type_a} ({round(type_a/wrong_count*100,1)}%) — wrong chunks retrieved")
        print(f"  Type B (Compression)  : {type_b} ({round(type_b/wrong_count*100,1)}%) — answer sentence dropped")
        print(f"  Type C (Model/LLM)    : {type_c} ({round(type_c/wrong_count*100,1)}%) — context ok, LLM wrong")

        insights = []
        if type_a >= type_b and type_a >= type_c:
            insights.append("Biggest problem: Retrieval — Hybrid RAG (BM25+vector) would help Type A")
        if type_b >= type_a and type_b >= type_c:
            insights.append("Biggest problem: Compression — increase sentence retention to help Type B")
        if type_c >= type_a and type_c >= type_b:
            insights.append("Biggest problem: LLM — prompt engineering or stronger model would help Type C")

        print(f"\nINSIGHTS:")
        for insight in insights:
            print(f"  - {insight}")

    print(f"\nOTHER METRICS:")
    print(f"  Queries retried         : {retried_count}/{len(rows)}")
    print(f"  Avg compression ratio   : {avg_compression}")
    print(f"  Avg confidence score    : {avg_confidence}")
    print(f"\nResults saved to:")
    print(f"  CSV : {output_csv}")
    print(f"  JSON: {output_json}")


if __name__ == "__main__":
    run_evaluation()
