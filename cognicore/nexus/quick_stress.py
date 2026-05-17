"""Quick summary of all 5 stress tests."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from cognicore.research.persistent_store import PersistentCognitionStore
from cognicore.nexus.run import run_nexus
from cognicore.nexus.trajectory_store import TrajectoryStore

def stats(v):
    n = len(v); m = sum(v)/n
    s = math.sqrt(sum((x-m)**2 for x in v)/(n-1)) if n > 1 else 0
    return m, s, 1.96*s/math.sqrt(n) if n > 1 else 0

print("=" * 60)
print("  TEST 1: MULTI-SEED STABILITY (3 seeds x 4 policies)")
print("=" * 60)
policies = ['minimal', 'standard', 'test_first', 'review_first']
R = {p: [] for p in policies}
for seed in [1, 42, 99]:
    for p in policies:
        PersistentCognitionStore().clear()
        r = run_nexus(policy=p, max_attempts=5, quiet=True)
        R[p].append(r['solved'])

print(f"  {'Policy':<15} {'Mean':>6} {'Std':>5} {'95CI':>6}")
print(f"  {'-'*35}")
for p in policies:
    m, s, c = stats(R[p])
    print(f"  {p:<15} {m:>5.1f} {s:>5.1f}  +/-{c:.1f}")

print()
print("=" * 60)
print("  TEST 2: COLD vs WARM MEMORY")
print("=" * 60)
PersistentCognitionStore().clear()
TrajectoryStore().clear()
cold = run_nexus(policy='standard', max_attempts=5, quiet=True)
warm = run_nexus(policy='standard', max_attempts=5, quiet=True)
ca = sum(x['attempts'] for x in cold['results'])
wa = sum(x['attempts'] for x in warm['results'])
print(f"  Cold: {cold['solved']}/20, attempts={ca}, tokens={cold['token_total']:,}")
print(f"  Warm: {warm['solved']}/20, attempts={wa}, tokens={warm['token_total']:,}")
print(f"  Attempt delta: {wa-ca:+d}")
print(f"  Memory learns: {'YES' if wa <= ca else 'NO'}")

print()
print("=" * 60)
print("  TEST 3: REVIEWER IMPACT")
print("=" * 60)
PersistentCognitionStore().clear()
no_r = run_nexus(policy='minimal', max_attempts=5, quiet=True)
PersistentCognitionStore().clear()
with_r = run_nexus(policy='standard', max_attempts=5, quiet=True)
nf = [x['task_id'] for x in no_r['results'] if not x['solved']]
wf = [x['task_id'] for x in with_r['results'] if not x['solved']]
print(f"  Without reviewer: {no_r['solved']}/20, failed={nf}")
print(f"  With reviewer:    {with_r['solved']}/20, failed={wf}")
extra_fails = set(wf) - set(nf)
if extra_fails:
    print(f"  Reviewer caused failures: {extra_fails}")
else:
    print(f"  Reviewer: no additional failures")
print(f"  Token overhead: {with_r['token_total'] - no_r['token_total']:+,}")

print()
print("=" * 60)
print("  TEST 4: TOKEN ECONOMICS")
print("=" * 60)
print(f"  {'Policy':<15} {'Solved':>7} {'Tokens':>10} {'Tok/Solve':>10}")
print(f"  {'-'*45}")
for p in policies:
    PersistentCognitionStore().clear()
    r = run_nexus(policy=p, max_attempts=5, quiet=True)
    tps = r['token_total'] / max(r['solved'], 1)
    print(f"  {p:<15} {r['solved']:>5}/20 {r['token_total']:>10,} {tps:>10,.0f}")

print()
print("=" * 60)
print("  TEST 5: LONG-SESSION DRIFT (3 runs)")
print("=" * 60)
PersistentCognitionStore().clear()
TrajectoryStore().clear()
for i in range(3):
    r = run_nexus(policy='standard', max_attempts=5, quiet=True)
    a = sum(x['attempts'] for x in r['results'])
    print(f"  Run {i+1}: {r['solved']}/20, attempts={a}, tokens={r['token_total']:,}")

ts = TrajectoryStore()
print(f"\n  Total trajectories: {ts.get_stats()['total_trajectories']}")
print("=" * 60)
