"""
SCENE 7 — LongMemEval Benchmark.
Runs the actual benchmark. Prints real timing and real R@K scores.
No numbers are hardcoded.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

print("=" * 60)
print("  LONGMEMEVAL — Multi-Hop Memory Benchmark")
print("=" * 60)
print()
print("  Testing: ZeroShot baseline vs CogniCore Multi-Hop Adapter")
print("  Metric: STRICT R@5 (exact match, top-5 recall)")
print()

t0 = time.time()

try:
    from cognicore_benchmarks.longmemeval.runner import run_benchmark
    results = run_benchmark()
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.1f}s\n")
    print(f"  {'Window':<10} {'ZeroShot':<14} {'Multi-Hop':<14} {'Gain'}")
    print("  " + "─" * 50)
    for row in results:
        gain = row['multihop'] - row['zeroshot']
        symbol = "🚀" if gain > 0 else "  "
        print(f"  w={row['window']:<8} {row['zeroshot']:.1%}{'':10} {row['multihop']:.1%}{'':10} {gain:+.1%} {symbol}")

except ImportError as e:
    # Fallback: run the runner script directly
    elapsed = time.time() - t0
    print(f"  Import error: {e}")
    print("  Running runner.py directly...")
    os.system("python cognicore_benchmarks/longmemeval/runner.py")
