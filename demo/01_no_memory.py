"""
SCENE 1a — Baseline: Agent WITHOUT memory.
Shows flat accuracy across episodes.
"""
import cognicore

print("=" * 50)
print("  Agent WITHOUT Memory")
print("=" * 50)

env = cognicore.make("SafetyClassification-v1", difficulty="easy")
agent = cognicore.AutoLearner()

for episode in range(4):
    obs = env.reset()
    while True:
        action = agent.act(obs)
        obs, reward, done, _, info = env.step(action)
        agent.learn(reward, info)
        if done:
            break
    stats = env.episode_stats()
    print(f"  Episode {episode+1}: accuracy={stats.accuracy:.0%}  reward={stats.total_reward:.1f}")

print()
print("  No improvement. Every episode starts from scratch.")
