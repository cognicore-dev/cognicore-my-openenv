#!/usr/bin/env python3
"""
NEXUS Runner — Multi-agent autonomous SWE benchmark.

Usage:
  python -m cognicore.nexus.run                    # standard (rule-based)
  python -m cognicore.nexus.run --llm               # real LLM agents
  python -m cognicore.nexus.run --policy test_first  # different routing
  python -m cognicore.nexus.run --compare            # compare all policies
  python -m cognicore.nexus.run --llm --compare      # compare with LLM
"""
import sys, os, argparse, json, time, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cognicore.runtime import CogniCoreRuntime, RuntimeConfig
from cognicore.research.swebench import load_swebench_tasks
from cognicore.research.persistent_store import PersistentCognitionStore
from cognicore.research.run_swebench import FIXES
from cognicore.nexus.agent import AgentRole, AgentContext, AgentRegistry
from cognicore.nexus.agents import (
    MemoryAgent, PlannerAgent, CoderAgent,
    ReviewerAgent, TesterAgent, VerifierAgent,
)
from cognicore.nexus.coordinator import CoordinationBus
from cognicore.nexus.token_tracker import TokenTracker
from cognicore.nexus.trajectory_store import TrajectoryStore, Trajectory

SESSION = uuid.uuid4().hex[:8]

def cprint(tag, msg, color="37"):
    COLORS = {"OK":"32","FAIL":"31","AGENT":"36","BUS":"35",
              "TOKEN":"33","INFO":"37","POLICY":"94","RESULT":"32"}
    c = COLORS.get(tag, color)
    print(f"  \033[{c}m[{tag}]\033[0m {msg}")


def build_registry(runtime=None, persistent=None, isolated=False, use_llm=False):
    """Build the full agent registry."""
    reg = AgentRegistry()
    reg.register(MemoryAgent(runtime=runtime, persistent_store=persistent))
    reg.register(PlannerAgent())
    if use_llm:
        from cognicore.nexus.llm_agents import LLMCoderAgent, LLMReviewerAgent
        from cognicore.research.llm_client import LLMClient
        llm = LLMClient(provider="auto")
        if llm.available:
            cprint("LLM", f"Using {llm.provider}/{llm.model} for code generation")
            reg.register(LLMCoderAgent(llm=llm))
            reg.register(LLMReviewerAgent(llm=llm))
        else:
            cprint("LLM", "No LLM available, falling back to rule-based")
            reg.register(CoderAgent(fix_db=FIXES))
            reg.register(ReviewerAgent(similarity_threshold=0.85))
    else:
        reg.register(CoderAgent(fix_db=FIXES))
        reg.register(ReviewerAgent(similarity_threshold=0.85))
    reg.register(TesterAgent(isolated=isolated))
    reg.register(VerifierAgent())
    return reg


def run_nexus(policy="standard", max_attempts=5, isolated=False, quiet=False,
              use_llm=False):
    """Run NEXUS multi-agent pipeline on all SWE-bench tasks."""
    tasks = load_swebench_tasks()
    runtime = CogniCoreRuntime(config=RuntimeConfig(
        reflection_min_samples=1, reflection_failure_threshold=1, memory_top_k=5,
    ), name=f"nexus-{SESSION}")
    persistent = PersistentCognitionStore()
    tracker = TokenTracker(budget_tokens=1000000, budget_usd=50.0)
    registry = build_registry(runtime, persistent, isolated, use_llm=use_llm)
    bus = CoordinationBus(registry, tracker, policy=policy, max_messages=15)
    traj_store = TrajectoryStore()

    results = []
    solved = 0

    if not quiet:
        print(f"\n{'='*72}")
        print(f"  NEXUS — Multi-Agent Autonomous SWE Benchmark")
        print(f"  Session: {SESSION} | Policy: {policy} | Tasks: {len(tasks)}")
        print(f"  Agents: {', '.join(a.role.value for a in registry.all())}")
        print(f"{'='*72}\n")

    for task in tasks:
        ctx = AgentContext(
            task_id=task.id,
            task_description=task.issue,
            buggy_code=task.buggy_code,
            test_code=task.test_code,
            metadata={"category": task.category},
        )

        t0 = time.perf_counter()
        success, events = bus.execute_pipeline(ctx, max_attempts=max_attempts)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if success:
            solved += 1

        # Collect per-task stats
        task_result = {
            "task_id": task.id,
            "category": task.category,
            "solved": success,
            "attempts": ctx.attempt,
            "agents_called": len(events),
            "elapsed_ms": elapsed_ms,
        }
        results.append(task_result)

        # Record trajectory for offline RL
        traj = Trajectory(task_id=task.id, category=task.category,
                          policy=policy, solved=success,
                          total_attempts=ctx.attempt, wall_clock_ms=elapsed_ms)
        for e in events:
            traj.add_step(agent=e.role.value, action=e.role.value,
                          tactic=e.result.artifacts.get('tactic',''),
                          tokens_in=e.result.tokens_in, tokens_out=e.result.tokens_out,
                          cost_usd=e.result.cost_usd, latency_ms=e.result.latency_ms,
                          success=e.result.success, error=e.result.error[:100])
        traj_store.store(traj)

        # Store to CogniCore memory
        last_tactic = "unknown"
        for e in reversed(events):
            if e.role == AgentRole.CODER:
                last_tactic = e.result.artifacts.get("tactic", "unknown")
                break

        runtime.memory.store({
            "category": task.category, "correct": success, "bug_id": task.id,
            "predicted": f"tactic:{last_tactic} {'PASS' if success else 'FAIL'}"
        })
        persistent.store_episode(SESSION, task.category, task.id,
                                  last_tactic, "PASS" if success else "FAIL",
                                  "", "", last_tactic, success)
        persistent.store_strategy(task.category, last_tactic, success)

        if not quiet:
            status = "\033[32mPASS\033[0m" if success else "\033[31mFAIL\033[0m"
            agents_used = [e.role.value[0].upper() for e in events]
            print(f"  {task.id:<18} {status}  A{ctx.attempt}  "
                  f"{'→'.join(agents_used)}  {elapsed_ms}ms")

    # ── REPORT ──
    total = len(tasks)
    task_tokens = tracker.per_task()

    print(f"\n{'='*72}")
    print(f"  NEXUS RESULTS — {policy} policy")
    print(f"{'='*72}")
    print(f"  Solved: {solved}/{total} ({100*solved/total:.1f}%)")
    print(f"  Total agent calls: {sum(r['agents_called'] for r in results)}")
    print(f"\n  Token Usage:")
    print(f"  {tracker.summary()}")
    print(f"\n  Agent Stats:")
    for role, stats in registry.stats().items():
        print(f"    {role:<12} {stats['calls']:>3} calls  {stats['tokens']:>8,} tokens  {stats['model']}")

    # Per-category breakdown
    cats = {}
    for r in results:
        c = r["category"]
        if c not in cats:
            cats[c] = {"total": 0, "solved": 0}
        cats[c]["total"] += 1
        cats[c]["solved"] += int(r["solved"])

    print(f"\n  Category Breakdown:")
    for cat, s in sorted(cats.items()):
        bar = "█" * s["solved"] + "░" * (s["total"] - s["solved"])
        print(f"    {cat:<18} {bar} {s['solved']}/{s['total']}")
    print(f"\n  Trajectory Store: {traj_store.get_stats()['total_trajectories']} trajectories")
    print(f"{'='*72}\n")

    return {"policy": policy, "solved": solved, "total": total, "results": results,
            "token_total": tracker.total_tokens, "cost_total": tracker.total_cost}


def compare_policies(max_attempts=5, use_llm=False):
    """Compare all routing policies head-to-head."""
    print(f"\n{'='*72}")
    print(f"  NEXUS POLICY COMPARISON")
    print(f"{'='*72}\n")

    all_results = {}
    for policy in ["minimal", "standard", "test_first", "review_first"]:
        # Clear persistent store between policies for fair comparison
        PersistentCognitionStore().clear()
        r = run_nexus(policy=policy, max_attempts=max_attempts, quiet=True,
                      use_llm=use_llm)
        all_results[policy] = r
        print(f"  {policy:<15} solved={r['solved']}/{r['total']}  "
              f"tokens={r['token_total']:>8,}  cost=${r['cost_total']:.4f}")

    print(f"\n  {'Policy':<15} {'Solve%':>7} {'Tokens':>10} {'Cost':>8} {'Calls':>7}")
    print(f"  {'-'*50}")
    for p, r in all_results.items():
        pct = 100 * r["solved"] / r["total"]
        calls = sum(x["agents_called"] for x in r["results"])
        print(f"  {p:<15} {pct:>6.1f}% {r['token_total']:>10,} ${r['cost_total']:>7.4f} {calls:>7}")

    best = max(all_results.items(), key=lambda x: x[1]["solved"])
    print(f"\n  Best policy: {best[0]} ({best[1]['solved']}/{best[1]['total']} solved)")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NEXUS Multi-Agent SWE Runner")
    p.add_argument("--policy", default="standard",
                   choices=["minimal", "standard", "test_first", "review_first"])
    p.add_argument("--attempts", type=int, default=5)
    p.add_argument("--isolated", action="store_true")
    p.add_argument("--llm", action="store_true", help="Use real LLM agents")
    p.add_argument("--compare", action="store_true", help="Compare all policies")
    a = p.parse_args()

    if a.compare:
        compare_policies(a.attempts, use_llm=a.llm)
    else:
        run_nexus(policy=a.policy, max_attempts=a.attempts,
                  isolated=a.isolated, use_llm=a.llm)
