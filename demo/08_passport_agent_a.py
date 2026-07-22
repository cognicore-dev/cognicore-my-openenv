"""
SCENE 6a — AgentPassport: Agent A trains, saves .passport file.
Run this first, then run 09_passport_agent_b.py
"""
import cognicore
from cognicore.passport import AgentPassport
from cognicore.replay import EventStore, EventRecorder
from cognicore.memory.sqlite_backend import SQLiteMemoryBackend

DB = "agent_a_memory.db"

config = cognicore.CogniCoreConfig(enable_memory=True, enable_reflection=True)
env = cognicore.make("SafetyClassification-v1", difficulty="easy", config=config)
agent = cognicore.AutoLearner()
store = EventStore()
recorder = EventRecorder(store=store)
memory = SQLiteMemoryBackend(DB)

print("=" * 55)
print("  AGENT A — Training + Passport Export")
print("=" * 55)
print()

for ep in range(4):
    obs = env.reset()
    recorder.record_simple(f"episode_{ep}", "episode_start", agent="agent_a")
    while True:
        action = agent.act(obs)
        obs, reward, done, _, info = env.step(action)
        agent.learn(reward, info)
        case_text = str(obs.get("case", obs.get("observation", "")))
        memory.add(case_text[:200], category="experience", scope="project")
        if done:
            break
    recorder.record_simple(f"episode_{ep}", "episode_end", agent="agent_a")
    stats = env.episode_stats()
    mem_count = len(memory.list())
    print(f"  Episode {ep+1}: accuracy={stats.accuracy:.0%}  memories={mem_count}")

print()
print(f"  Saving passport...")
passport_path = AgentPassport.checkpoint(agent, env, store, memory)
print(f"  ✅ Passport: {passport_path}")
print(f"  Total memory entries: {len(memory.list())}")
print()
print("  >>> Now run 09_passport_agent_b.py <<<")
