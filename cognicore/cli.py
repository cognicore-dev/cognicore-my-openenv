"""
CogniCore CLI — Train, evaluate, and benchmark from the command line.

Usage:
    python -m cognicore.cli train --env MazeRunner-v0 --algo PPO --steps 100000
    python -m cognicore.cli benchmark --env GridWorld-v0
    python -m cognicore.cli list
    python -m cognicore.cli arena --envs MazeRunner-v0,GridWorld-v0
    python -m cognicore.cli ui   # Start NEXUS dashboard
"""
import argparse
import sys
import time


def cmd_list(args):
    import cognicore.gym
    import gymnasium as gym
    print("\n  CogniCore Gymnasium Environments:")
    print("  " + "-" * 45)
    gym_envs = sorted(e for e in gym.envs.registry.keys() if e.startswith("cognicore/"))
    for eid in gym_envs:
        print(f"    {eid}")
    print(f"\n  Total: {len(gym_envs)} gymnasium-native envs")


def cmd_train(args):
    import cognicore.gym
    import gymnasium as gym
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy

    env_id = args.env if args.env.startswith("cognicore/") else f"cognicore/{args.env}"
    algos = {"ppo": PPO, "dqn": DQN, "a2c": A2C}
    algo_cls = algos.get(args.algo.lower())
    if not algo_cls:
        print(f"  Unknown algo: {args.algo}. Choose: {list(algos.keys())}")
        return

    print(f"\n  Training {args.algo.upper()} on {env_id} for {args.steps:,} steps...")
    env = Monitor(gym.make(env_id))
    t0 = time.time()
    model = algo_cls("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=args.steps)
    dt = time.time() - t0
    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=args.eval_episodes)
    print(f"  Done in {dt:.1f}s | Score: {mean_r:+.1f} +/- {std_r:.1f}")
    if args.save:
        model.save(args.save)
        print(f"  Saved to {args.save}")
    env.close()


def cmd_benchmark(args):
    import cognicore.gym
    import gymnasium as gym
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    import numpy as np

    env_id = args.env if args.env.startswith("cognicore/") else f"cognicore/{args.env}"
    print(f"\n  Benchmarking {env_id} ({args.steps:,} steps each)...")
    print(f"  {'Algo':<10} {'Score':>10} {'Std':>8} {'Time':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    env = Monitor(gym.make(env_id))
    rr = []
    for _ in range(20):
        o, _ = env.reset(); t, d = 0, False
        while not d:
            o, r, te, tr, _ = env.step(env.action_space.sample())
            t += r; d = te or tr
        rr.append(t)
    print(f"  {'Random':<10} {np.mean(rr):>+9.1f} {np.std(rr):>7.1f}        -")
    env.close()

    for name, cls in [("PPO", PPO), ("DQN", DQN), ("A2C", A2C)]:
        env = Monitor(gym.make(env_id))
        t0 = time.time()
        model = cls("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=args.steps)
        dt = time.time() - t0
        mr, sr = evaluate_policy(model, env, n_eval_episodes=20)
        print(f"  {name:<10} {mr:>+9.1f} {sr:>7.1f} {dt:>7.1f}s")
        env.close()


def cmd_arena(args):
    import cognicore as cc
    env_ids = [e.strip() for e in args.envs.split(",")]
    arena = cc.Arena()
    acts = ["UP","DOWN","LEFT","RIGHT","HOLD","BUY","SELL",
            "FORAGE","HUNT","BUILD_SHELTER","CRAFT_TOOL","REST","EXPLORE","DEFEND"]
    arena.add_agent("Random", cc.RandomAgent())
    arena.add_agent("Q-Learning", cc.QLearningAgent(acts, epsilon_decay=0.97))
    arena.add_agent("SARSA", cc.SARSAAgent(acts, epsilon_decay=0.97))
    arena.add_agent("Bandit", cc.BanditAgent(acts))
    arena.add_agent("Genetic", cc.GeneticAgent(acts, population_size=8))
    print(f"\n  Running arena on: {env_ids}")
    arena.run_tournament(env_ids, episodes_per_match=args.episodes)
    arena.print_leaderboard()


def cmd_ui(args):
    from cognicore.ui.server import start_server
    port = getattr(args, 'port', 7842)
    start_server(port=port, open_browser=True)


def cmd_integrations(args):
    action = getattr(args, 'action', 'status')
    if action == 'setup':
        from cognicore.integrations.setup_wizard import setup_wizard
        setup_wizard()
    elif action == 'test':
        from cognicore.integrations.setup_wizard import test_connections
        test_connections()
    elif action == 'status':
        from cognicore.integrations.task_queue import NexusTaskQueue
        q = NexusTaskQueue()
        s = q.stats()
        print("\n  NEXUS Integration Status")
        print(f"  Tasks: {s['total']} | Dead letter: {s['dead_letter']}")
        print(f"  By status: {s['by_status']}")
        print(f"  By source: {s['by_source']}\n")


def cmd_webhooks(args):
    from cognicore.ui.server import start_server
    port = getattr(args, 'port', 7842)
    print("  Starting webhook server (includes dashboard)...")
    start_server(port=port, open_browser=False)


def main():
    p = argparse.ArgumentParser(prog="cognicore")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("list", help="List environments")
    t = sub.add_parser("train", help="Train agent")
    t.add_argument("--env", required=True); t.add_argument("--algo", default="PPO")
    t.add_argument("--steps", type=int, default=50000)
    t.add_argument("--eval-episodes", type=int, default=20)
    t.add_argument("--save", default=None)
    b = sub.add_parser("benchmark", help="Benchmark algos")
    b.add_argument("--env", required=True); b.add_argument("--steps", type=int, default=50000)
    a = sub.add_parser("arena", help="ELO tournament")
    a.add_argument("--envs", required=True); a.add_argument("--episodes", type=int, default=20)
    u = sub.add_parser("ui", help="Start NEXUS dashboard")
    u.add_argument("--port", type=int, default=7842)
    i = sub.add_parser("integrations", help="Manage integrations")
    i.add_argument("action", nargs="?", default="status", choices=["setup", "test", "status"])
    w = sub.add_parser("webhooks", help="Start webhook server")
    w.add_argument("--port", type=int, default=7842)
    args = p.parse_args()
    cmds = {"list": cmd_list, "train": cmd_train, "benchmark": cmd_benchmark,
            "arena": cmd_arena, "ui": cmd_ui, "integrations": cmd_integrations,
            "webhooks": cmd_webhooks}
    if args.command in cmds:
        cmds[args.command](args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
