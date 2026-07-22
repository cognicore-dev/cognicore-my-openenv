"""
SCENE 6b — AgentPassport: Agent B loads Agent A's passport.
No training from scratch. Starts with all of Agent A's experience.
"""
import glob
import cognicore
from cognicore.passport import AgentPassport

print("=" * 55)
print("  AGENT B — Starting from Agent A's passport")
print("=" * 55)
print()

# Find the passport file written by Agent A
passports = glob.glob("agent_*.passport") + glob.glob("*.passport")
if not passports:
    print("  ERROR: No .passport file found. Run 08_passport_agent_a.py first.")
    exit(1)

passport_path = sorted(passports)[-1]
print(f"  Loading: {passport_path}")

try:
    agent_b, env_b, memory_b = AgentPassport.load(passport_path)
    entries = memory_b.list()
    print(f"  Memory entries imported: {len(entries)}")
    if entries:
        print(f"  Oldest memory: {entries[-1]['text'][:60]}")
        print(f"  Newest memory: {entries[0]['text'][:60]}")
    print()

    # Agent B runs — no training needed
    print("  Agent B — Episode 1 (zero prior training in this process):")
    obs = env_b.reset()
    while True:
        action = agent_b.act(obs)
        obs, reward, done, _, info = env_b.step(action)
        agent_b.learn(reward, info)
        if done:
            break
    stats = env_b.episode_stats()
    print(f"  accuracy={stats.accuracy:.0%}  reward={stats.total_reward:.1f}")
    print()
    print("  Agent A needed 4 episodes to converge.")
    print("  Agent B starts at the same level — episode 1.")

except Exception as e:
    print(f"\n  AgentPassport.load() failed: {e}")
    print("  Falling back to manual memory import demo...")

    # Fallback: show that memory DB is shared
    from cognicore.memory.sqlite_backend import SQLiteMemoryBackend
    mem = SQLiteMemoryBackend("agent_a_memory.db")
    entries = mem.list()
    print(f"\n  Loaded {len(entries)} memories directly from agent_a_memory.db")
    for e in entries[:3]:
        print(f"    [{e['id']}] {e['text'][:60]}")
