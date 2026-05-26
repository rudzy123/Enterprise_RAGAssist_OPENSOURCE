#!/usr/bin/env python3
"""
Run end-to-end RAG evaluations over all questions in a JSONL file.

Captures retrieval metrics plus answer quality (citations, groundedness),
saves timestamped JSON results, and prints summary metrics.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from answer_generation.generation import generate_answer_from_chunks
from evals.metrics import aggregate_metrics, compute_answer_metrics, compute_retrieval_metrics
from retrieval.retrieve_chunks import retrieve_similar_chunks

RESULTS_DIR = Path("evals")
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_FINAL_K_VALUES = [2, 3, 4, 5]
DEFAULT_MIN_SIMILARITY_VALUES = [0.35, 0.40, 0.45, 0.50, 0.55]


def load_questions(jsonl_path: str) -> list:
    questions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def evaluate_question(
    question_obj: dict,
    *,
    final_k: int,
    min_similarity: float,
    retrieve_k: int,
    rerank_enabled: bool,
    use_llm: bool,
    verbose: bool,
) -> dict:
    question_text = question_obj.get("question", "")
    start = time.time()

    trace = retrieve_similar_chunks(
        question_text,
        retrieve_k=retrieve_k,
        final_k=final_k,
        min_similarity=min_similarity,
        rerank_enabled=rerank_enabled,
        return_trace=True,
        structured_logs=False,
    )

    retrieval_latency_ms = (time.time() - start) * 1000
    final_chunks = trace["chunks"]
    ranked_chunks = trace.get("reranked") or trace.get("threshold_passed") or final_chunks
    raw_candidate_count = len(trace.get("raw_candidates") or [])
    abstained = len(final_chunks) == 0

    answer, confidence, confidence_reason, _, _ = generate_answer_from_chunks(
        question_text,
        final_chunks,
        use_llm=use_llm,
        raw_candidate_count=raw_candidate_count,
    )

    retrieval_metrics = compute_retrieval_metrics(
        question_obj,
        final_chunks=final_chunks,
        ranked_chunks=ranked_chunks,
        k=final_k,
        abstained=abstained,
        latency_ms=retrieval_latency_ms,
    )
    answer_metrics = compute_answer_metrics(
        question_obj,
        answer=answer,
        chunks=final_chunks,
        abstained=abstained,
        confidence=confidence,
        confidence_reason=confidence_reason,
    )
    metrics = {**retrieval_metrics, **answer_metrics}

    if verbose:
        print(f"\nQuestion: {question_text}")
        print(f"Expected: {question_obj.get('source_doc_id')}")
        print(f"Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")
        print(f"Citations: {'yes' if metrics['has_citations'] else 'no'}")
        print(f"Grounded: {'yes' if metrics['grounded'] else 'no'} ({metrics['groundedness_reason']})")
        print(f"Returned: {metrics['returned_sources']} (chunks={metrics['num_retrieved_chunks']})")
        print(
            f"Hit@k: {'✓' if metrics['hit_at_k'] else '✗'}  "
            f"P@{final_k}: {metrics['precision_at_k']:.2f}  "
            f"RR: {metrics['reciprocal_rank']:.2f}  "
            f"conf: {metrics['confidence']:.2f}"
        )

    return {
        "question": question_text,
        "expected_source": question_obj.get("source_doc_id"),
        "metrics": metrics,
        "config": {
            "final_k": final_k,
            "min_chunk_similarity": min_similarity,
            "retrieve_k": retrieve_k,
            "rerank_enabled": rerank_enabled,
            "use_llm": use_llm,
        },
    }


def run_single_config(
    questions: list,
    *,
    final_k: int,
    min_similarity: float,
    retrieve_k: int,
    rerank_enabled: bool,
    use_llm: bool,
    verbose: bool,
) -> dict:
    results = [
        evaluate_question(
            q,
            final_k=final_k,
            min_similarity=min_similarity,
            retrieve_k=retrieve_k,
            rerank_enabled=rerank_enabled,
            use_llm=use_llm,
            verbose=verbose,
        )
        for q in questions
    ]
    summary = aggregate_metrics([r["metrics"] for r in results])
    summary["config"] = {
        "final_k": final_k,
        "min_chunk_similarity": min_similarity,
        "retrieve_k": retrieve_k,
        "rerank_enabled": rerank_enabled,
        "use_llm": use_llm,
    }
    return {
        "run_at": datetime.now().isoformat(),
        "num_questions": len(questions),
        "summary": summary,
        "results": results,
    }


def print_summary(summary: dict) -> None:
    print(f"Total questions:     {summary['total_questions']}")
    print(f"% with citations:    {summary['pct_with_citations'] * 100:.1f}%")
    print(f"% grounded:          {summary['pct_grounded'] * 100:.1f}%")
    print(f"Avg groundedness:    {summary['avg_groundedness_score']:.3f}")
    print(f"Avg confidence:      {summary['avg_confidence']:.3f}")
    print(f"Hit rate:            {summary['hit_rate']:.3f}")
    print(f"Hit@k:               {summary['hit_at_k']:.3f}")
    print(f"Precision@k:         {summary['precision_at_k']:.3f}")
    print(f"MRR:                 {summary['mrr']:.3f}")
    print(f"Abstention rate:     {summary['abstention_rate']:.3f}")
    print(f"Avg latency ms:      {summary['avg_latency_ms']:.1f}")


def run_parameter_sweep(
    questions: list,
    *,
    final_k_values: list[int],
    min_similarity_values: list[float],
    retrieve_k: int,
    rerank_enabled: bool,
    use_llm: bool,
    verbose: bool,
) -> dict:
    sweep_runs = []
    total = len(final_k_values) * len(min_similarity_values)
    run_idx = 0

    for final_k in final_k_values:
        for min_similarity in min_similarity_values:
            run_idx += 1
            print(f"\n{'=' * 80}")
            print(f"SWEEP {run_idx}/{total}: final_k={final_k}, min_similarity={min_similarity}")
            print("=" * 80)

            run = run_single_config(
                questions,
                final_k=final_k,
                min_similarity=min_similarity,
                retrieve_k=retrieve_k,
                rerank_enabled=rerank_enabled,
                use_llm=use_llm,
                verbose=verbose,
            )
            print_summary(run["summary"])
            sweep_runs.append(run)

    best = max(
        sweep_runs,
        key=lambda r: (
            r["summary"]["pct_grounded"],
            r["summary"]["pct_with_citations"],
            r["summary"]["mrr"],
            r["summary"]["precision_at_k"],
            -r["summary"]["abstention_rate"],
        ),
    )

    return {
        "run_at": datetime.now().isoformat(),
        "sweep": True,
        "num_questions": len(questions),
        "final_k_values": final_k_values,
        "min_similarity_values": min_similarity_values,
        "runs": [
            {"summary": r["summary"], "num_results": len(r["results"])} for r in sweep_runs
        ],
        "best_config": best["summary"]["config"],
        "best_summary": best["summary"],
    }


def save_results(payload: dict, prefix: str = "results") -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_path = RESULTS_DIR / f"{prefix}_{timestamp}.json"
    latest_path = RESULTS_DIR / f"{prefix}.json"

    for path in (timestamp_path, latest_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return timestamp_path, latest_path


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluations over JSONL questions")
    parser.add_argument(
        "jsonl_path",
        nargs="?",
        default="evals/questions.jsonl",
        help="Path to questions JSONL file",
    )
    parser.add_argument("--sweep", action="store_true", help="Sweep final_k and min similarity")
    parser.add_argument("--final-k", type=int, default=None, help="Final k (overrides config)")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="Minimum chunk similarity threshold",
    )
    parser.add_argument("--retrieve-k", type=int, default=15, help="Bi-encoder retrieve_k")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking during eval",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use retrieval-only answer generation (no OpenAI)",
    )
    parser.add_argument(
        "--final-k-values",
        type=str,
        default=",".join(str(v) for v in DEFAULT_FINAL_K_VALUES),
        help="Comma-separated final_k values for sweep",
    )
    parser.add_argument(
        "--min-similarity-values",
        type=str,
        default=",".join(str(v) for v in DEFAULT_MIN_SIMILARITY_VALUES),
        help="Comma-separated min similarity values for sweep",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Per-question output")
    args = parser.parse_args()

    try:
        questions = load_questions(args.jsonl_path)
    except FileNotFoundError:
        print(f"Error: Could not find {args.jsonl_path}")
        sys.exit(1)

    if not questions:
        print("No questions loaded.")
        sys.exit(1)

    print(f"Loaded {len(questions)} questions from {args.jsonl_path}")

    rerank_enabled = not args.no_rerank
    use_llm = not args.no_llm and bool(os.getenv("OPENAI_API_KEY"))
    if not args.no_llm and not use_llm:
        print("No OPENAI_API_KEY set — using retrieval-only answer generation.")

    if args.sweep:
        final_k_values = [int(x.strip()) for x in args.final_k_values.split(",") if x.strip()]
        min_similarity_values = [
            float(x.strip()) for x in args.min_similarity_values.split(",") if x.strip()
        ]
        payload = run_parameter_sweep(
            questions,
            final_k_values=final_k_values,
            min_similarity_values=min_similarity_values,
            retrieve_k=args.retrieve_k,
            rerank_enabled=rerank_enabled,
            use_llm=use_llm,
            verbose=args.verbose,
        )
        print("\n" + "=" * 80)
        print("BEST CONFIG")
        print("=" * 80)
        print(json.dumps(payload["best_config"], indent=2))
        print_summary(payload["best_summary"])
        ts_path, latest_path = save_results(payload, prefix="sweep_results")
    else:
        from config import FINAL_K, MIN_CHUNK_SIMILARITY

        final_k = args.final_k if args.final_k is not None else FINAL_K
        min_similarity = (
            args.min_similarity if args.min_similarity is not None else MIN_CHUNK_SIMILARITY
        )

        print(
            f"\nConfig: final_k={final_k}, min_similarity={min_similarity}, "
            f"rerank={rerank_enabled}, use_llm={use_llm}"
        )
        print("=" * 80)

        payload = run_single_config(
            questions,
            final_k=final_k,
            min_similarity=min_similarity,
            retrieve_k=args.retrieve_k,
            rerank_enabled=rerank_enabled,
            use_llm=use_llm,
            verbose=args.verbose,
        )

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print_summary(payload["summary"])
        ts_path, latest_path = save_results(payload)

    print(f"\n✅ Results saved to:")
    print(f"- {ts_path}")
    print(f"- {latest_path}")


if __name__ == "__main__":
    main()
