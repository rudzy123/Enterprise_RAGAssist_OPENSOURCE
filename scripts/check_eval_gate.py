#!/usr/bin/env python3
"""
Evaluate retrieval quality gates for CI/CD.

Usage:
    # Check an existing results file
    python scripts/check_eval_gate.py evals/results.json

    # Run evals first, then check (no LLM, no rerank for speed)
    python scripts/check_eval_gate.py --run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _ensure_path() -> None:
    """Add project root to sys.path for local package imports."""
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_summary(path: Path) -> dict:
    _ensure_path()
    from evals.metrics import aggregate_metrics

    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary") or payload
    if "hit_at_k" not in summary and "results" in payload:
        summary = aggregate_metrics([r["metrics"] for r in payload["results"]])
    return summary


def _check_thresholds(summary: dict) -> list[str]:
    """Return list of failed gate messages (empty == pass)."""
    _ensure_path()
    from core.config import (
        EVAL_MAX_ABSTENTION_RATE,
        EVAL_MIN_HIT_AT_K,
        EVAL_MIN_MRR,
        EVAL_MIN_PCT_GROUNDED,
        EVAL_MIN_PRECISION_AT_K,
    )

    failures = []

    checks = [
        ("hit_at_k", summary.get("hit_at_k", 0.0), EVAL_MIN_HIT_AT_K, ">="),
        ("mrr", summary.get("mrr", 0.0), EVAL_MIN_MRR, ">="),
        ("precision_at_k", summary.get("precision_at_k", 0.0), EVAL_MIN_PRECISION_AT_K, ">="),
        ("pct_grounded", summary.get("pct_grounded", 0.0), EVAL_MIN_PCT_GROUNDED, ">="),
        ("abstention_rate", summary.get("abstention_rate", 1.0), EVAL_MAX_ABSTENTION_RATE, "<="),
    ]

    for name, actual, threshold, op in checks:
        if op == ">=" and actual < threshold:
            failures.append(f"{name}: {actual:.3f} < required {threshold:.3f}")
        elif op == "<=" and actual > threshold:
            failures.append(f"{name}: {actual:.3f} > max allowed {threshold:.3f}")

    return failures


def main() -> int:
    _ensure_path()
    from core.config import (
        EVAL_MAX_ABSTENTION_RATE,
        EVAL_MIN_HIT_AT_K,
        EVAL_MIN_MRR,
        EVAL_MIN_PCT_GROUNDED,
        EVAL_MIN_PRECISION_AT_K,
    )

    parser = argparse.ArgumentParser(description="Check RAG eval quality gates")
    parser.add_argument(
        "results_path",
        nargs="?",
        default="evals/results.json",
        help="Path to eval results JSON",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run evals/run_evals.py --no-llm --no-rerank before checking",
    )
    args = parser.parse_args()

    results_path = ROOT / args.results_path

    if args.run:
        print("Running evaluations (no LLM, no rerank)...")
        cmd = [
            sys.executable,
            str(ROOT / "evals" / "run_evals.py"),
            "--no-llm",
            "--no-rerank",
        ]
        env = {**os.environ.copy(), "PYTHONPATH": str(ROOT)}
        result = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
        if result.returncode != 0:
            print("Eval run failed.", file=sys.stderr)
            return result.returncode

    if not results_path.exists():
        print(f"Results file not found: {results_path}", file=sys.stderr)
        return 1

    summary = _load_summary(results_path)
    failures = _check_thresholds(summary)

    print("Eval gate thresholds:")
    print(f"  hit_at_k          >= {EVAL_MIN_HIT_AT_K}")
    print(f"  mrr               >= {EVAL_MIN_MRR}")
    print(f"  precision_at_k    >= {EVAL_MIN_PRECISION_AT_K}")
    print(f"  pct_grounded      >= {EVAL_MIN_PCT_GROUNDED}")
    print(f"  abstention_rate   <= {EVAL_MAX_ABSTENTION_RATE}")
    print()
    print("Actual results:")
    for key in ("hit_at_k", "mrr", "precision_at_k", "pct_grounded", "abstention_rate"):
        val = summary.get(key)
        if val is not None:
            print(f"  {key}: {val:.3f}")

    if failures:
        print("\nFAILED quality gates:", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    print("\nAll quality gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
