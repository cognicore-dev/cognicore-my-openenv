"""
CogniCore Benchmark Regression Checker.

Compares current benchmark results against a saved baseline and
flags regressions. Returns exit code 1 if any metric regresses
beyond the configured thresholds.

Usage::

    # Generate baseline
    python -m cognicore.benchmarks.suite --output benchmark_output/baseline.json

    # Check for regressions
    python -m cognicore.benchmarks.regression --baseline benchmark_output/baseline.json
"""

from __future__ import annotations

import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger("cognicore.benchmarks.regression")

# ── Thresholds ───────────────────────────────────────────────────────
ACCURACY_DROP_THRESHOLD = 0.02      # Flag if accuracy drops > 2%
SOLVE_RATE_DROP_THRESHOLD = 0.05    # Flag if solve rate drops > 5%
LATENCY_INCREASE_THRESHOLD = 0.20   # Flag if latency increases > 20%


def check_regression(
    current_path: str,
    baseline_path: str,
    accuracy_threshold: float = ACCURACY_DROP_THRESHOLD,
    latency_threshold: float = LATENCY_INCREASE_THRESHOLD,
) -> Tuple[bool, List[str]]:
    """Compare current results against baseline.

    Parameters
    ----------
    current_path : str
        Path to current benchmark results JSON.
    baseline_path : str
        Path to baseline benchmark results JSON.
    accuracy_threshold : float
        Maximum acceptable accuracy drop (0.02 = 2%).
    latency_threshold : float
        Maximum acceptable latency increase (0.20 = 20%).

    Returns
    -------
    tuple of (passed, messages)
        passed is True if no regressions detected.
        messages is a list of human-readable findings.
    """
    baseline = _load_json(baseline_path)
    current = _load_json(current_path)

    if not baseline or not current:
        return False, ["Could not load benchmark files."]

    messages = []
    passed = True

    # ── Compare env results ──────────────────────────────────────
    baseline_envs = _index_env_results(baseline.get("env_results", []))
    current_envs = _index_env_results(current.get("env_results", []))

    for key, b_data in baseline_envs.items():
        if key not in current_envs:
            messages.append(f"  ⚠ Missing: {key} not in current results")
            continue

        c_data = current_envs[key]
        b_acc = b_data.get("avg_accuracy", 0)
        c_acc = c_data.get("avg_accuracy", 0)

        if b_acc > 0 and (b_acc - c_acc) > accuracy_threshold:
            passed = False
            messages.append(
                f"  ✗ REGRESSION: {key} accuracy dropped "
                f"{b_acc:.1%} → {c_acc:.1%} (Δ{c_acc - b_acc:+.1%})"
            )
        else:
            messages.append(f"  ✓ {key}: accuracy {c_acc:.1%} (baseline {b_acc:.1%})")

    # ── Compare backend latency ──────────────────────────────────
    baseline_backends = {
        b["backend_name"]: b for b in baseline.get("backend_results", [])
    }
    current_backends = {
        b["backend_name"]: b for b in current.get("backend_results", [])
    }

    for name, b_data in baseline_backends.items():
        if name not in current_backends:
            continue

        c_data = current_backends[name]
        for metric in ["store_avg_ms", "search_avg_ms", "category_avg_ms"]:
            b_val = b_data.get(metric, 0)
            c_val = c_data.get(metric, 0)

            if b_val > 0 and c_val > 0:
                increase = (c_val - b_val) / b_val
                if increase > latency_threshold:
                    passed = False
                    messages.append(
                        f"  ✗ LATENCY: {name}.{metric} increased "
                        f"{b_val:.2f}ms → {c_val:.2f}ms (+{increase:.0%})"
                    )

    return passed, messages


def _load_json(path: str) -> Dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to load %s: %s", path, e)
        return {}


def _index_env_results(results: List[Dict]) -> Dict[str, Dict]:
    """Index env results by (env_id, difficulty, condition) key."""
    indexed = {}
    for r in results:
        eps = r.get("episodes", 0)
        if eps == 0:
            continue
        key = f"{r['env_id']}/{r['difficulty']}/{r['condition']}"
        # Compute avg if not present
        r["avg_accuracy"] = r.get("total_accuracy", 0) / eps
        r["avg_reward"] = r.get("total_reward", 0) / eps
        indexed[key] = r
    return indexed


def _cli_main():
    import argparse

    parser = argparse.ArgumentParser(description="CogniCore Benchmark Regression Checker")
    parser.add_argument("--baseline", required=True, help="Path to baseline JSON")
    parser.add_argument("--current", default=None, help="Path to current JSON (or run fresh)")
    parser.add_argument("--accuracy-threshold", type=float, default=ACCURACY_DROP_THRESHOLD)
    parser.add_argument("--latency-threshold", type=float, default=LATENCY_INCREASE_THRESHOLD)
    args = parser.parse_args()

    # If no current file, run a quick benchmark
    if not args.current:
        from cognicore.benchmarks.suite import BenchmarkSuite

        print("\n  No current results file — running quick benchmark...")
        suite = BenchmarkSuite()
        result = suite.run(quick=True)
        current_path = "benchmark_output/_current_check.json"
        result.to_json(current_path)
        args.current = current_path

    print(f"\n  Checking regressions: {args.current} vs {args.baseline}")
    print(f"  Thresholds: accuracy={args.accuracy_threshold:.0%}, latency={args.latency_threshold:.0%}\n")

    passed, messages = check_regression(
        args.current, args.baseline,
        args.accuracy_threshold, args.latency_threshold,
    )

    for msg in messages:
        print(msg)

    if passed:
        print("\n  ✓ No regressions detected.\n")
        sys.exit(0)
    else:
        print("\n  ✗ Regressions detected! See above.\n")
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
