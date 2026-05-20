import sys; sys.path.insert(0, '.')
from cognicore.nexus.autonomous import NexusRunner

def log(step):
    ph = step["phase"].upper().ljust(7)
    st = "OK" if step["status"] == "done" else "FAIL" if step["status"] == "failed" else ".."
    print(f"  [{ph}] [{st}] {step['action']}")
    if step.get("detail"):
        for line in step["detail"].split("\n")[:4]:
            print(f"           {line}")

runner = NexusRunner(max_attempts=2)
runner.on_step(log)

print("\n  NEXUS Autonomous Runner — Live Test")
print("  ====================================\n")

result = runner.solve(
    "Fix detect_encoding crash when content is None",
    repo_path=".",
    auto_pr=False
)

print(f"\n  === RESULT ===")
print(f"  Solved: {result.solved}")
print(f"  Attempts: {result.attempts}")
print(f"  Files: {result.files_changed}")
print(f"  Tests: {result.tests_passed}P / {result.tests_failed}F")
print(f"  Duration: {result.duration}s")
if result.patch:
    print(f"  Patch preview: {result.patch[:200]}")
print()
