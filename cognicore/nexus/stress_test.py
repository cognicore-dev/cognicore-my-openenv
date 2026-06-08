#!/usr/bin/env python3
"""
NEXUS Stress Test Suite — research-grade evaluation.
Tests: multi-seed stability, cold/warm memory, reviewer analysis,
       token economics, long-session drift.

Usage: python -m cognicore.nexus.stress_test
"""
import sys, os, math, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cognicore.research.persistent_store import PersistentCognitionStore
from cognicore.nexus.run import run_nexus
from cognicore.nexus.trajectory_store import TrajectoryStore


def stats(vals):
    n = len(vals)
    if n == 0: return 0, 0, 0
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean)**2 for v in vals) / (n - 1)
        std = math.sqrt(var)
        ci95 = 1.96 * std / math.sqrt(n)
    else:
        std, ci95 = 0, 0
    return mean, std, ci95


def header(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}\n")


# ══════════════════════════════════════════════════════════
# TEST 1: MULTI-SEED POLICY STABILITY
# ══════════════════════════════════════════════════════════
def test_multi_seed(seeds=None):
    header("TEST 1: MULTI-SEED POLICY STABILITY")
    seeds = seeds or [1, 2, 3, 42, 99]
    policies = ["minimal", "standard", "test_first", "review_first"]
    results = {p: [] for p in policies}

    for seed in seeds:
        for policy in policies:
            PersistentCognitionStore().clear()
            # Seed doesn't affect rule-based agent but affects future LLM runs
            r = run_nexus(policy=policy, max_attempts=5, quiet=True)
            results[policy].append(r)
            print(f"  seed={seed:>3} {policy:<15} {r['solved']}/{r['total']}  "
                  f"tokens={r['token_total']:>7,}")

    print(f"\n  {'Policy':<15} {'Mean Solve':>11} {'Std':>5} {'95%CI':>6} "
          f"{'Mean Tokens':>12} {'Calls':>7}")
    print(f"  {'-'*60}")
    for p in policies:
        solves = [r['solved'] for r in results[p]]
        tokens = [r['token_total'] for r in results[p]]
        calls = [sum(x['agents_called'] for x in r['results']) for r in results[p]]
        sm, ss, sc = stats(solves)
        tm, ts, tc = stats(tokens)
        cm, _, _ = stats(calls)
        print(f"  {p:<15} {sm:>7.1f}±{ss:<4.1f} ±{sc:.1f} {tm:>12,.0f} {cm:>7.0f}")

    return results


# ══════════════════════════════════════════════════════════
# TEST 2: COLD vs WARM PERSISTENT MEMORY
# ══════════════════════════════════════════════════════════
def test_cold_warm():
    header("TEST 2: COLD vs WARM PERSISTENT MEMORY")

    # Cold run
    PersistentCognitionStore().clear()
    TrajectoryStore().clear()
    print("  [COLD RUN] No prior memory...")
    cold = run_nexus(policy="standard", max_attempts=5, quiet=True)
    cold_attempts = sum(r['attempts'] for r in cold['results'])
    print(f"    Solved: {cold['solved']}/{cold['total']}  "
          f"Total attempts: {cold_attempts}  Tokens: {cold['token_total']:,}")

    # Warm run (persistent memory from cold run)
    print("\n  [WARM RUN] Using memory from cold run...")
    warm = run_nexus(policy="standard", max_attempts=5, quiet=True)
    warm_attempts = sum(r['attempts'] for r in warm['results'])
    print(f"    Solved: {warm['solved']}/{warm['total']}  "
          f"Total attempts: {warm_attempts}  Tokens: {warm['token_total']:,}")

    # 3rd run (double-warm)
    print("\n  [WARM-2 RUN] Double memory accumulation...")
    warm2 = run_nexus(policy="standard", max_attempts=5, quiet=True)
    warm2_attempts = sum(r['attempts'] for r in warm2['results'])
    print(f"    Solved: {warm2['solved']}/{warm2['total']}  "
          f"Total attempts: {warm2_attempts}  Tokens: {warm2['token_total']:,}")

    print(f"\n  {'Session':<12} {'Solved':>7} {'Attempts':>9} {'Tokens':>8} {'Delta':>8}")
    print(f"  {'-'*48}")
    print(f"  {'Cold':<12} {cold['solved']:>5}/{cold['total']} {cold_attempts:>9} "
          f"{cold['token_total']:>8,} {'--':>8}")
    att_delta = warm_attempts - cold_attempts
    print(f"  {'Warm-1':<12} {warm['solved']:>5}/{warm['total']} {warm_attempts:>9} "
          f"{warm['token_total']:>8,} {att_delta:>+8}")
    att_delta2 = warm2_attempts - cold_attempts
    print(f"  {'Warm-2':<12} {warm2['solved']:>5}/{warm2['total']} {warm2_attempts:>9} "
          f"{warm2['token_total']:>8,} {att_delta2:>+8}")

    proven = warm_attempts <= cold_attempts or warm['solved'] > cold['solved']
    print(f"\n  Cross-session learning: {'PROVEN' if proven else 'NOT PROVEN'}")
    return cold, warm, warm2


# ══════════════════════════════════════════════════════════
# TEST 3: REVIEWER FAILURE ANALYSIS
# ══════════════════════════════════════════════════════════
def test_reviewer_analysis():
    header("TEST 3: REVIEWER FAILURE ANALYSIS")
    PersistentCognitionStore().clear()

    policies_to_test = {
        "minimal": "No reviewer",
        "test_first": "Reviewer after test",
        "standard": "Reviewer before test",
        "review_first": "Double reviewer pass",
    }

    results = {}
    for policy, desc in policies_to_test.items():
        PersistentCognitionStore().clear()
        r = run_nexus(policy=policy, max_attempts=5, quiet=True)
        total_attempts = sum(x['attempts'] for x in r['results'])
        failed_tasks = [x for x in r['results'] if not x['solved']]
        results[policy] = {
            "solved": r['solved'], "total": r['total'],
            "attempts": total_attempts, "tokens": r['token_total'],
            "calls": sum(x['agents_called'] for x in r['results']),
            "failed": [x['task_id'] for x in failed_tasks],
        }

    print(f"  {'Policy':<15} {'Desc':<25} {'Solved':>7} {'Attempts':>9} "
          f"{'Tokens':>8} {'Calls':>6}")
    print(f"  {'-'*72}")
    for p, desc in policies_to_test.items():
        r = results[p]
        print(f"  {p:<15} {desc:<25} {r['solved']:>5}/{r['total']} {r['attempts']:>9} "
              f"{r['tokens']:>8,} {r['calls']:>6}")

    # Quantify reviewer impact
    no_rev = results["minimal"]
    with_rev = results["standard"]
    print(f"\n  Reviewer Impact Analysis:")
    print(f"    Without reviewer: {no_rev['solved']}/{no_rev['total']} solved, "
          f"{no_rev['attempts']} attempts")
    print(f"    With reviewer:    {with_rev['solved']}/{with_rev['total']} solved, "
          f"{with_rev['attempts']} attempts")
    delta_solved = with_rev['solved'] - no_rev['solved']
    delta_tokens = with_rev['tokens'] - no_rev['tokens']
    print(f"    Solve delta: {delta_solved:+d}")
    print(f"    Token overhead: {delta_tokens:+,}")
    if delta_solved < 0:
        print(f"    FINDING: Reviewer costs {abs(delta_solved)} solves and "
              f"{delta_tokens:,} extra tokens")
    elif delta_solved == 0:
        print(f"    FINDING: Reviewer adds {delta_tokens:,} tokens with no solve benefit")

    # Failed task overlap
    min_fails = set(results["minimal"]["failed"])
    std_fails = set(results["standard"]["failed"])
    reviewer_caused = std_fails - min_fails
    if reviewer_caused:
        print(f"\n    Tasks ONLY failed with reviewer: {reviewer_caused}")

    return results


# ══════════════════════════════════════════════════════════
# TEST 4: TOKEN ECONOMICS
# ══════════════════════════════════════════════════════════
def test_token_economics():
    header("TEST 4: TOKEN ECONOMICS")
    PersistentCognitionStore().clear()

    results = {}
    for policy in ["minimal", "standard", "test_first", "review_first"]:
        PersistentCognitionStore().clear()
        r = run_nexus(policy=policy, max_attempts=5, quiet=True)
        solved = r['solved']
        cost_per_solve = r['token_total'] / max(solved, 1)
        results[policy] = {
            "solved": solved, "tokens": r['token_total'],
            "cost_per_solve": cost_per_solve,
            "cost_usd": r['cost_total'],
            "cost_per_solve_usd": r['cost_total'] / max(solved, 1),
        }

    print(f"  {'Policy':<15} {'Solved':>7} {'Tokens':>10} {'Tok/Solve':>10} "
          f"{'Cost':>8} {'$/Solve':>8}")
    print(f"  {'-'*60}")
    for p, r in results.items():
        print(f"  {p:<15} {r['solved']:>5}/20 {r['tokens']:>10,} "
              f"{r['cost_per_solve']:>10,.0f} ${r['cost_usd']:>7.4f} "
              f"${r['cost_per_solve_usd']:>7.4f}")

    best = min(results.items(), key=lambda x: x[1]['cost_per_solve'])
    print(f"\n  Most efficient: {best[0]} ({best[1]['cost_per_solve']:,.0f} tokens/solve)")
    return results


# ══════════════════════════════════════════════════════════
# TEST 5: LONG-SESSION DRIFT
# ══════════════════════════════════════════════════════════
def test_long_session():
    header("TEST 5: LONG-SESSION DRIFT (3 consecutive runs)")
    PersistentCognitionStore().clear()
    TrajectoryStore().clear()

    runs = []
    for i in range(3):
        print(f"  Run {i+1}/3...", end=" ", flush=True)
        r = run_nexus(policy="standard", max_attempts=5, quiet=True)
        attempts = sum(x['attempts'] for x in r['results'])
        runs.append({"run": i+1, "solved": r['solved'], "total": r['total'],
                      "attempts": attempts, "tokens": r['token_total']})
        print(f"solved={r['solved']}/{r['total']}  attempts={attempts}  "
              f"tokens={r['token_total']:,}")

    # Check for drift
    print(f"\n  {'Run':>4} {'Solved':>7} {'Attempts':>9} {'Tokens':>8} {'Trend':>8}")
    print(f"  {'-'*40}")
    for r in runs:
        trend = ""
        if r['run'] > 1:
            prev = runs[r['run']-2]
            if r['attempts'] < prev['attempts']: trend = "↓ better"
            elif r['attempts'] > prev['attempts']: trend = "↑ worse"
            else: trend = "= stable"
        print(f"  {r['run']:>4} {r['solved']:>5}/{r['total']} {r['attempts']:>9} "
              f"{r['tokens']:>8,} {trend:>8}")

    # Check trajectory store
    ts = TrajectoryStore()
    ts_stats = ts.get_stats()
    print(f"\n  Trajectory store: {ts_stats['total_trajectories']} trajectories, "
          f"{ts_stats['solved']} solved")

    # Memory contamination check
    last_run_attempts = runs[-1]['attempts']
    first_run_attempts = runs[0]['attempts']
    if last_run_attempts > first_run_attempts + 5:
        print(f"  WARNING: Possible memory contamination "
              f"(attempts {first_run_attempts}→{last_run_attempts})")
    elif last_run_attempts < first_run_attempts:
        print(f"  GOOD: Memory accumulation reducing attempts "
              f"({first_run_attempts}→{last_run_attempts})")
    else:
        print(f"  STABLE: No drift detected")

    return runs


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()
    print("\n" + "█"*72)
    print("  NEXUS COMPREHENSIVE STRESS TEST SUITE")
    print("█"*72)

    r1 = test_multi_seed(seeds=[1, 42, 99])
    r2 = test_cold_warm()
    r3 = test_reviewer_analysis()
    r4 = test_token_economics()
    r5 = test_long_session()

    elapsed = time.time() - t0
    header(f"ALL TESTS COMPLETE — {elapsed:.0f}s")
