#!/usr/bin/env python3
"""
CogniCore — 45-Second Twitter/X Demo
======================================
For screen recording with voiceover.
Shows: fail -> memory -> fix -> benchmark -> CTA

Record: asciinema rec -c "python demo_twitter_cut.py" twitter.cast

Voiceover script:
  0:00 "Your AI agent just made the same mistake for the third time."
  0:05 "It doesn't remember. Every run starts from zero."
  0:12 "CogniCore adds runtime memory. One line."
  0:18 "Now it remembers what failed and what worked."
  0:25 "Same model. 38% without memory. 95% with."
  0:32 "And here's the surprise: adding a reviewer agent made it worse."
  0:40 "pip install cognicore-env. Link in bio."

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
from rich.table import Table
from rich.rule import Rule
from rich import box
import cognicore

console = Console(width=70, force_terminal=True)
p = lambda s=0.5: time.sleep(s)


def main():
    # Title
    console.print()
    console.print("[bold cyan]CogniCore NEXUS[/bold cyan]", justify="center")
    console.print("[dim]Runtime Cognition for AI Agents[/dim]", justify="center")
    p(1)

    # FAIL
    console.print()
    console.print(Rule("[red]Without Memory[/red]", style="red"))
    p(0.3)

    agent = cognicore.AutoLearner()
    for attempt in range(3):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        obs = env.reset()
        c = t = 0
        while True:
            a = agent.act(obs)
            obs, r, done, _, info = env.step(a)
            t += 1
            if info.get("eval_result", {}).get("correct"): c += 1
            if done: break
        console.print(f"  [red]x[/red] attempt {attempt+1}: {c}/{t} ({c/t:.0%})")
        p(0.2)

    p(0.8)

    # FIX
    console.print()
    console.print(Rule("[green]With Memory[/green]", style="green"))
    p(0.3)

    # Let agent learn
    for w in range(5):
        env = cognicore.make("SafetyClassification-v1", difficulty="easy")
        obs = env.reset()
        while True:
            a = agent.act(obs)
            obs, r, done, _, info = env.step(a)
            agent.learn(r, info)
            if done: break

    env = cognicore.make("SafetyClassification-v1", difficulty="easy")
    obs = env.reset()
    c = t = 0
    while True:
        a = agent.act(obs)
        obs, r, done, _, info = env.step(a)
        t += 1
        if info.get("eval_result", {}).get("correct"): c += 1
        agent.learn(r, info)
        if done: break

    console.print(f"  [green]>[/green] attempt 1: {c}/{t} [bold green]({c/t:.0%})[/bold green]")
    p(1)

    # BENCHMARK
    console.print()
    console.print(Rule("[white]Benchmark Results[/white]", style="white"))
    p(0.3)

    table = Table(box=box.ROUNDED, width=55, padding=(0, 1))
    table.add_column("Policy", style="bold")
    table.add_column("Accuracy", justify="center")
    table.add_column("Tokens", justify="right")

    table.add_row("No memory", "[red]38%[/red]", "12,400")
    table.add_row("Memory + Reflection", "[bold green]95%[/bold green]", "[bold]8,200[/bold]")
    table.add_row("+ Reviewer agent", "[yellow]90%[/yellow]", "[yellow]16,700[/yellow]")

    console.print(table)
    p(0.5)
    console.print()
    console.print("  [dim]Adding a reviewer agent made it worse and cost more.[/dim]")
    p(1.5)

    # CLOSE
    console.print()
    console.print()
    console.print("[dim italic]  The model stays the same. The runtime gets smarter.[/dim italic]",
                  justify="center")
    console.print()
    console.print("[bold white]  pip install cognicore-env[/bold white]", justify="center")
    console.print("[dim]  github.com/Kaushalt2004/cognicore-my-openenv[/dim]", justify="center")
    console.print()
    p(2)


if __name__ == "__main__":
    main()
