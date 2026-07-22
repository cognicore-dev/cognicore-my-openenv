"""
SCENE 1b — Agent WITH memory and reflection.
Shows improving accuracy across episodes.
"""
import cognicore

print("=" * 50)
print("  Agent WITH CogniCore Memory + Reflection")
print("=" * 50)

config = cognicore.CogniCoreConfig(enable_memory=True, enable_reflection=True)
env = cognicore.make("SafetyClassification-v1", difficulty="easy", config=config)
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
print("  Memory-enabled: each episode learns from the last.")
