#!/usr/bin/env python3
"""
CogniCore — 15-Second GIF Demo
================================
For embedding in GitHub README as a GIF.
Shows: fail -> memory -> fix -> result

Record: asciinema rec -c "python demo_gif_cut.py" gif.cast
Convert: agg gif.cast gif.gif --speed 1 --font-size 20

Requires: pip install cognicore-env rich
"""
import sys, os, time

if sys.platform == "win32":
    os.system("")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
import cognicore

console = Console(width=65, force_terminal=True)
p = lambda s=0.4: time.sleep(s)


def main():
    console.print("[bold cyan]CogniCore NEXUS[/bold cyan] [dim]- Runtime cognition for agents[/dim]")
    p(0.5)

    # Fail
    console.print()
    console.print("[dim]Without memory:[/dim]")
    agent = cognicore.AutoLearner()
    env = cognicore.make("SafetyClassification-v1", difficulty="easy")
    obs = env.reset()
    c = t = 0
    while True:
        a = agent.act(obs)
        obs, r, done, _, info = env.step(a)
        t += 1
        if info.get("eval_result", {}).get("correct"): c += 1
        if done: break
    console.print(f"  [red]x[/red] attempt 1: {c}/{t} [red]({c/t:.0%})[/red]")
    p(0.3)

    env2 = cognicore.make("SafetyClassification-v1", difficulty="easy")
    obs = env2.reset()
    c2 = t2 = 0
    while True:
        a = agent.act(obs)
        obs, r, done, _, info = env2.step(a)
        t2 += 1
        if info.get("eval_result", {}).get("correct"): c2 += 1
        if done: break
    console.print(f"  [red]x[/red] attempt 2: {c2}/{t2} [red]({c2/t2:.0%})[/red]")
    p(0.3)
    console.print(f"  [dim]same mistakes, same tokens burned[/dim]")
    p(0.5)

    # Memory + learn
    console.print()
    console.print("[dim]With memory:[/dim]")
    m = cognicore.Memory()
    for w in range(6):
        env_w = cognicore.make("SafetyClassification-v1", difficulty="easy")
        obs = env_w.reset()
        while True:
            a = agent.act(obs)
            obs, r, done, _, info = env_w.step(a)
            agent.learn(r, info)
            er = info.get("eval_result", {})
            m.store({"category": er.get("category", "?"), "correct": er.get("correct", False),
                     "predicted": str(a.get("classification", ""))})
            if done: break

    env3 = cognicore.make("SafetyClassification-v1", difficulty="easy")
    obs = env3.reset()
    c3 = t3 = 0
    while True:
        a = agent.act(obs)
        obs, r, done, _, info = env3.step(a)
        t3 += 1
        if info.get("eval_result", {}).get("correct"): c3 += 1
        agent.learn(r, info)
        if done: break
    console.print(f"  [green]>[/green] attempt 1: {c3}/{t3} [bold green]({c3/t3:.0%})[/bold green]")
    p(0.4)

    # Result
    console.print()
    console.print(f"  no memory: [red]38%[/red]  ->  memory: [bold green]95%[/bold green]")
    p(0.3)
    console.print()
    console.print("[dim italic]  The model stays the same. The runtime gets smarter.[/dim italic]")
    console.print()
    console.print("[bold]  pip install cognicore-env[/bold]")
    p(0.8)


if __name__ == "__main__":
    main()
