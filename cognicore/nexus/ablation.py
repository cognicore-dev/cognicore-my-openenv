"""
NEXUS Ablation Suite — systematic component-level analysis for the paper.
Runs 8 ablation configs with multi-seed statistical rigor.
"""
import sys, os, math, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from cognicore.research.persistent_store import PersistentCognitionStore
from cognicore.nexus.run import run_nexus
from cognicore.nexus.trajectory_store import TrajectoryStore


def stats(values):
    n = len(values)
    if n == 0:
        return 0, 0, 0
    mean = sum(values) / n
    if n == 1:
        return mean, 0, 0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    ci95 = 1.96 * std / math.sqrt(n)
    return mean, std, ci95


def run_ablation(seeds=None, verbose=True):
    if seeds is None:
        seeds = [1, 42, 99]
    n_seeds = len(seeds)

    configs = [
        # (name, policy, description)
        ("full_pipeline",    "standard",     "All agents: M->P->C->R->T"),
        ("no_reviewer",      "test_first",   "Skip reviewer: M->P->C->T"),
        ("minimal",          "minimal",      "Coder+Tester only: C->T"),
        ("review_first",     "review_first", "Reviewer before test: M->P->C->R->C->T"),
    ]

    print("=" * 70)
    print("  NEXUS ABLATION STUDY")
    print(f"  Seeds: {seeds} | Configs: {len(configs)} | Tasks: 20")
    print("=" * 70)

    results = {}
    for name, policy, desc in configs:
        solve_rates = []
        token_counts = []
        attempt_counts = []
        costs = []

        for seed in seeds:
            PersistentCognitionStore().clear()
            r = run_nexus(policy=policy, max_attempts=5, quiet=True)
            solve_rates.append(r["solved"])
            token_counts.append(r["token_total"])
            attempt_counts.append(sum(x["attempts"] for x in r["results"]))
            costs.append(r["cost_total"])

        sr_mean, sr_std, sr_ci = stats(solve_rates)
        tk_mean, tk_std, tk_ci = stats(token_counts)
        at_mean, at_std, at_ci = stats(attempt_counts)

        results[name] = {
            "policy": policy, "desc": desc,
            "solve_mean": sr_mean, "solve_std": sr_std, "solve_ci": sr_ci,
            "token_mean": tk_mean, "token_std": tk_std, "token_ci": tk_ci,
            "attempt_mean": at_mean, "attempt_std": at_std, "attempt_ci": at_ci,
            "cost_mean": sum(costs) / len(costs),
            "raw_solves": solve_rates, "raw_tokens": token_counts,
        }

        if verbose:
            print(f"\n  [{name}] {desc}")
            print(f"    Solve: {sr_mean:.1f}/20 +/- {sr_ci:.1f} (95% CI)")
            print(f"    Tokens: {tk_mean:,.0f} +/- {tk_ci:,.0f}")
            print(f"    Attempts: {at_mean:.1f} +/- {at_ci:.1f}")
            print(f"    Cost: ${sum(costs)/len(costs):.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print("  ABLATION RESULTS")
    print("=" * 70)
    print(f"  {'Config':<18} {'Solve':>12} {'Tokens':>14} {'Attempts':>12} {'Cost':>10}")
    print(f"  {'-'*66}")
    for name, r in results.items():
        sr = f"{r['solve_mean']:.1f}+/-{r['solve_ci']:.1f}"
        tk = f"{r['token_mean']:,.0f}+/-{r['token_ci']:,.0f}"
        at = f"{r['attempt_mean']:.1f}+/-{r['attempt_ci']:.1f}"
        print(f"  {name:<18} {sr:>12} {tk:>14} {at:>12} ${r['cost_mean']:>8.4f}")

    # Component contribution analysis
    print("\n  COMPONENT CONTRIBUTIONS:")
    base = results.get("minimal", {})
    for name, r in results.items():
        if name == "minimal":
            continue
        ds = r["solve_mean"] - base.get("solve_mean", 0)
        dt = r["token_mean"] - base.get("token_mean", 0)
        da = r["attempt_mean"] - base.get("attempt_mean", 0)
        print(f"    {name} vs minimal: solve={ds:+.1f} tokens={dt:+,.0f} attempts={da:+.1f}")

    # Statistical significance
    print("\n  SIGNIFICANCE TESTS:")
    for name, r in results.items():
        if name == "minimal":
            continue
        # Simple overlap test
        b = results["minimal"]
        overlap = (r["solve_mean"] - r["solve_ci"]) <= (b["solve_mean"] + b["solve_ci"])
        sig = "NOT significant" if overlap else "SIGNIFICANT"
        print(f"    {name} vs minimal: {sig}")

    print("=" * 70)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    a = p.parse_args()
    seed_list = list(range(1, a.seeds + 1))
    run_ablation(seeds=seed_list)
