#!/usr/bin/env python3
"""
Analyze evaluation or sweep JSON results.
"""

import argparse
import json
from pathlib import Path


def load_results(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_single_run(results: list) -> dict:
    metrics_list = [r["metrics"] for r in results]
    total = len(metrics_list)
    if total == 0:
        return {}

    def avg(key: str) -> float:
        return sum(m[key] for m in metrics_list) / total

    worst_mrr = sorted(results, key=lambda r: r["metrics"].get("reciprocal_rank", 0.0))[:10]
    abstentions = [r for r in results if r["metrics"].get("abstained")]

    return {
        "total_cases": total,
        "pct_with_citations": avg("has_citations") if "has_citations" in metrics_list[0] else None,
        "pct_grounded": avg("grounded") if "grounded" in metrics_list[0] else None,
        "avg_groundedness_score": avg("groundedness_score")
        if "groundedness_score" in metrics_list[0]
        else None,
        "precision_at_k": avg("precision_at_k"),
        "mrr": sum(m["reciprocal_rank"] for m in metrics_list) / total,
        "abstention_rate": sum(1 for m in metrics_list if m["abstained"]) / total,
        "hit_rate": avg("retrieval_hit"),
        "worst_mrr": [
            {
                "question": r["question"],
                "reciprocal_rank": r["metrics"].get("reciprocal_rank"),
                "returned_sources": r["metrics"].get("returned_sources"),
                "abstained": r["metrics"].get("abstained"),
            }
            for r in worst_mrr
        ],
        "abstention_examples": [
            {
                "question": r["question"],
                "expected_source": r.get("expected_source"),
            }
            for r in abstentions[:10]
        ],
    }


def summarize_sweep(payload: dict) -> dict:
    runs = payload.get("runs", [])
    ranked = sorted(
        runs,
        key=lambda r: (
            r["summary"]["mrr"],
            r["summary"]["precision_at_k"],
            -r["summary"]["abstention_rate"],
        ),
        reverse=True,
    )
    return {
        "sweep": True,
        "num_configs": len(runs),
        "best_config": payload.get("best_config"),
        "best_summary": payload.get("best_summary"),
        "top_configs": [
            {
                "config": r["summary"]["config"],
                "mrr": r["summary"]["mrr"],
                "precision_at_k": r["summary"]["precision_at_k"],
                "abstention_rate": r["summary"]["abstention_rate"],
                "hit_at_k": r["summary"]["hit_at_k"],
            }
            for r in ranked[:5]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze eval JSON results")
    parser.add_argument("result_file", type=Path, help="Path to evals/results*.json")
    args = parser.parse_args()

    payload = load_results(args.result_file)

    if payload.get("sweep"):
        summary = summarize_sweep(payload)
    elif "results" in payload:
        summary = summarize_single_run(payload["results"])
    else:
        summary = summarize_single_run(payload)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
