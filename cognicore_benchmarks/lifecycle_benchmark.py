import time
import argparse
from typing import Dict

from cognicore.memory.tfidf_backend import TFIDFMemoryBackend
from cognicore.memory.lifecycle import MemoryLifecycleManager
from cognicore.memory.evaluator import RuleBasedEvaluator
from cognicore.memory.planner import RuleBasedPlanner

def run_benchmark(num_entries: int) -> Dict[str, float]:
    """Run lifecycle operations on a large number of simulated entries to measure throughput."""
    
    backend = TFIDFMemoryBackend()
    manager = MemoryLifecycleManager(
        backend=backend,
        evaluator=RuleBasedEvaluator(),
        planner=RuleBasedPlanner()
    )
    
    start_time = time.time()
    for i in range(num_entries):
        text = f"Simulated memory text {i} about fixing error."
        manager.observe(text, context={"task": f"task_{i}"}, source_component="benchmark")
    observe_time = time.time() - start_time
    
    start_time = time.time()
    for i in range(min(num_entries, 1000)):
        manager.retrieve(f"error task_{i}", top_k=5)
    retrieve_time = time.time() - start_time
    
    start_time = time.time()
    manager.run_lifecycle_pass()
    lifecycle_time = time.time() - start_time
    
    return {
        "observe_throughput": num_entries / max(0.001, observe_time),
        "retrieve_throughput": min(num_entries, 1000) / max(0.001, retrieve_time),
        "lifecycle_throughput": num_entries / max(0.001, lifecycle_time),
        "total_entries": backend.count()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark memory lifecycle operations.")
    parser.add_argument("--num_entries", type=int, default=1000, help="Number of entries to simulate.")
    args = parser.parse_args()
    
    print(f"Running benchmark with {args.num_entries} entries...")
    results = run_benchmark(args.num_entries)
    
    print("\nBenchmark Results:")
    print(f"Total Entries Stored: {results['total_entries']}")
    print(f"Observation Throughput: {results['observe_throughput']:.2f} entries/sec")
    print(f"Retrieval Throughput: {results['retrieve_throughput']:.2f} queries/sec")
    print(f"Lifecycle Pass Throughput: {results['lifecycle_throughput']:.2f} entries/sec")
