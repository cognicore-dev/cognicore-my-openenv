#!/usr/bin/env python3
"""
CogniCore NEXUS — Minimal GIF Demo (10-15 seconds)
For GitHub README badge/GIF. Tight output, no panels.
pip install cognicore-env rich
"""
import sys, os, io, time

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rich.console import Console

console = Console(width=72)
p = lambda s=0.35: time.sleep(s)


def main():
    import cognicore

    # Install
    console.print("[dim]$[/] [bold]pip install cognicore-env[/]")
    p(0.3)
    console.print("[green]✓ cognicore-env installed[/]")
    p(0.4)

    # Memory
    console.print("\n[bold cyan]>>> Memory: learning from mistakes[/]")
    p(0.3)

    m = cognicore.Memory()
    for pred in ["wrong_fix", "bad_patch", "retry_loop"]:
        m.store({"category": "login-bug", "correct": False, "predicted": pred})
        console.print(f"  [red]✗[/] stored failure: [dim]{pred}[/]")
        p(0.15)

    s = m.stats()
    console.print(f"  [yellow]→[/] memory: {s['total_entries']} entries, "
                  f"{s['failures']} failures, {s['success_rate']:.0%} success rate")
    p(0.4)

    # Reflection
    console.print("\n[bold cyan]>>> Reflection: analyzing patterns[/]")
    p(0.3)
    ref = cognicore.ReflectionEngine(m, min_samples=1, failure_threshold=1)
    hint = ref.get_hint("login-bug")
    if hint:
        console.print(f"  [yellow]💡 {hint}[/]")
    p(0.4)

    # Benchmark
    console.print("\n[bold cyan]>>> Benchmark: agent improvement[/]")
    p(0.3)

    import random
    random.seed(42)
    config = cognicore.CogniCoreConfig(enable_memory=True, enable_reflection=True)
    env = cognicore.make("SafetyClassification-v1", config=config)
    agent = cognicore.AutoLearner()

    scores = []
    for ep in range(20):
        obs = env.reset()
        while True:
            action = agent.act(obs)
            obs, reward, done, _, info = env.step(action)
            if hasattr(agent, "learn"):
                agent.learn(reward, info)
            if done:
                scores.append(env.episode_stats().accuracy)
                break

    early = sum(scores[:5]) / 5
    late = sum(scores[-5:]) / 5
    bar_early = "█" * int(early * 20) + "░" * (20 - int(early * 20))
    bar_late  = "█" * int(late * 20)  + "░" * (20 - int(late * 20))
    console.print(f"  ep 1-5:  [red]{bar_early} {early:.0%}[/]")
    console.print(f"  ep 16-20:[green]{bar_late} {late:.0%}[/]")
    console.print(f"  [bold bright_green]▲ +{(late - early)*100:.0f}% improvement with memory + reflection[/]")
    p(0.5)

    # Final
    console.print(f"\n[bold green]✓ cognicore-env v{cognicore.__version__}[/] "
                  f"— [dim]github.com/Kaushalt2004/cognicore-my-openenv[/]")
    p(0.5)


if __name__ == "__main__":
    main()
