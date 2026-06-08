"""Export trajectory data and show system status."""
import sys,os;sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','..'))
from cognicore.nexus.trajectory_store import TrajectoryStore

ts = TrajectoryStore()
stats = ts.get_stats()
print("TRAJECTORY STORE:")
print("  Total:", stats["total_trajectories"])
print("  Solved:", stats["solved"])
print("  Policies:", stats["policies"])

cp = ts.compare_policies()
print("\nPOLICY COMPARISON (from trajectory DB):")
for p, d in cp.items():
    n = d["n"]
    s = d["solved"]
    r = d["mean_reward"]
    t = d["avg_tokens"]
    print("  %-15s n=%3d solved=%3d reward=%+.3f avg_tok=%8.0f" % (p, n, s, r, t))

path = ts.export_for_training()
print("\nExported to:", path)
lines = open(path).readlines()
print("JSONL lines:", len(lines))
print("\nSample trajectory (first line):")
import json
first = json.loads(lines[0])
print("  task:", first["task_id"])
print("  policy:", first["policy"])
print("  solved:", first["solved"])
print("  reward:", first["total_reward"])
print("  steps:", len(first["steps"]))
