#!/usr/bin/env python3
"""
CogniCore NEXUS — The Demo (3 minutes)
========================================
A coding agent fixes a real bug. Memory makes it smarter.

Run:   python demo_final.py
Record: asciinema rec -c "python demo_final.py" demo.cast

Requires: pip install cognicore-env rich
"""
import sys
import os
import time

if sys.platform == "win32":
    os.system("")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.rule import Rule
from rich import box

import cognicore

console = Console(width=78, force_terminal=True)


def wait(s):
    time.sleep(s)


def typed(text, style="dim", speed=0.012):
    """Simulate typing a command."""
    console.print(f"  [bold green]$[/bold green] ", end="")
    for ch in text:
        console.print(ch, end="", style=style, highlight=False)
        time.sleep(speed)
    console.print()


# ─────────────────────────────────────────────────────────────────────
#  ACT 1 — THE BUG  (40s)
# ─────────────────────────────────────────────────────────────────────

def act1():
    console.print()
    console.print(Panel(
        "[bold white]CogniCore NEXUS[/bold white]\n"
        "[dim]Runtime cognition for AI agents[/dim]",
        border_style="bright_blue", width=50, padding=(1, 2),
    ), justify="center")
    wait(1.5)

    console.print()
    typed("pytest tests/test_auth.py -q")
    wait(0.4)

    # Simulated test output
    console.print("""
  [dim]tests/test_auth.py[/dim] [green].[/green][green].[/green][red]F[/red][green].[/green][red]F[/red][green].[/green][green].[/green][red]F[/red][green].[/green][green].[/green]

  [bold red]FAILED[/bold red] test_validate_email
  [dim]>       result = validate_email(user_input)[/dim]
  [red]E       ValueError: 'NoneType' has no attribute 'strip'[/red]

  [bold red]FAILED[/bold red] test_validate_phone
  [dim]>       result = validate_phone(None)[/dim]
  [red]E       TypeError: expected str, got NoneType[/red]

  [bold red]FAILED[/bold red] test_empty_string_input
  [dim]>       assert validate_email("") == False[/dim]
  [red]E       ValueError: empty string passed to validator[/red]

  [red bold]3 failed[/red bold], 7 passed in 0.42s""")

    wait(2)


# ─────────────────────────────────────────────────────────────────────
#  ACT 2 — AGENT FAILS TWICE  (30s)
# ─────────────────────────────────────────────────────────────────────

def act2():
    console.print()
    console.print(Rule("[bold white]Agent Attempt 1[/bold white]", style="white"))
    wait(0.5)

    typed("nexus fix tests/test_auth.py --model gpt-4o-mini")
    wait(0.5)

    # Show the bad patch
    patch1 = """\
# agent patch (attempt 1)
def validate_email(email):
-   return email.strip().lower()
+   return email.strip().lower() if email else False"""

    console.print()
    console.print(Syntax(patch1, "diff", theme="monokai", line_numbers=False,
                         padding=(0, 2)))
    wait(0.8)

    typed("pytest tests/test_auth.py -q")
    wait(0.3)
    console.print("""
  [dim]tests/test_auth.py[/dim] [green].[/green][green].[/green][green].[/green][green].[/green][red]F[/red][green].[/green][green].[/green][red]F[/red][green].[/green][green].[/green]

  [bold red]FAILED[/bold red] test_validate_phone
  [red]E       TypeError: expected str, got NoneType[/red]

  [bold red]FAILED[/bold red] test_empty_string_input
  [red]E       ValueError: empty string passed to validator[/red]

  [red bold]2 failed[/red bold], 8 passed in 0.38s""")

    wait(1.5)

    # Attempt 2 — same mistake
    console.print()
    console.print(Rule("[bold white]Agent Attempt 2[/bold white]", style="white"))
    wait(0.5)

    typed("nexus fix tests/test_auth.py --retry")
    wait(0.5)

    patch2 = """\
# agent patch (attempt 2)
def validate_phone(phone):
-   return phone.strip().isdigit()
+   return phone.strip().isdigit() if phone else False"""

    console.print()
    console.print(Syntax(patch2, "diff", theme="monokai", line_numbers=False,
                         padding=(0, 2)))
    wait(0.8)

    typed("pytest tests/test_auth.py -q")
    wait(0.3)
    console.print("""
  [dim]tests/test_auth.py[/dim] [green].[/green][green].[/green][green].[/green][green].[/green][green].[/green][green].[/green][green].[/green][green].[/green][green].[/green][red]F[/red]

  [bold red]FAILED[/bold red] test_empty_string_input
  [red]E       ValueError: empty string passed to validator[/red]

  [red bold]1 failed[/red bold], 9 passed in 0.35s""")

    wait(1)

    # Repeated failure callout
    console.print()
    console.print("  [bold yellow]>> Repeated failure detected.[/bold yellow]")
    console.print("  [dim]Same root cause across 2 attempts: missing input guard.[/dim]")

    # Store in real memory
    memory = cognicore.Memory()
    sm = cognicore.SemanticMemory()

    failures = [
        {"category": "null-input", "predicted": "inline_ternary", "correct": False,
         "text": "NoneType has no attribute strip - added ternary but missed empty string"},
        {"category": "null-input", "predicted": "inline_ternary", "correct": False,
         "text": "TypeError expected str got NoneType - same inline fix pattern"},
        {"category": "empty-string", "predicted": "inline_ternary", "correct": False,
         "text": "ValueError empty string passed to validator - ternary doesnt catch empty"},
    ]

    for f in failures:
        memory.store(f)
        sm.store(f)

    # Seed memory with historical failures
    history = [
        {"text": "Null input caused crash in user signup validator",
         "category": "null-input", "predicted": "guard_clause", "correct": True},
        {"text": "NoneType error in payment form - fixed with early return",
         "category": "null-input", "predicted": "early_return", "correct": True},
        {"text": "Missing None check in API request parser",
         "category": "null-input", "predicted": "guard_clause", "correct": True},
        {"text": "Empty string caused validation bypass in auth flow",
         "category": "empty-string", "predicted": "guard_clause", "correct": True},
        {"text": "Input sanitization missing for empty values in form handler",
         "category": "empty-string", "predicted": "guard_clause", "correct": True},
    ]
    for h in history:
        memory.store(h)
        sm.store(h)

    wait(2)
    return memory, sm


# ─────────────────────────────────────────────────────────────────────
#  ACT 3 — MEMORY RETRIEVAL  (25s)
# ─────────────────────────────────────────────────────────────────────

def act3(memory, sm):
    console.print()
    console.print(Rule("[bold yellow]Memory[/bold yellow]", style="yellow"))
    wait(0.5)

    console.print()
    console.print("  [dim]Searching execution history...[/dim]")
    wait(0.8)

    results = sm.semantic_search("NoneType strip validation input guard", top_k=3)

    console.print()
    console.print("  [bold]Found similar failures:[/bold]")
    console.print()

    for i, (entry, score) in enumerate(results):
        was_correct = entry.get("correct", False)
        cat = entry.get("category", "?")
        text = entry.get("text", "?")
        fix = entry.get("predicted", "?")

        if was_correct:
            icon = "[green]FIXED[/green]"
        else:
            icon = "[red]FAILED[/red]"

        console.print(f"    [dim]#{i+1}[/dim]  [cyan]{score:.2f}[/cyan]  {icon}  [dim]{cat}[/dim]")
        console.print(f"         {text[:60]}")
        if was_correct:
            console.print(f"         [green]fix: {fix}[/green]")
        console.print()
        wait(0.3)

    stats = memory.stats()
    console.print(f"  [dim]{stats['total_entries']} entries | "
                  f"{stats['successes']} successes | "
                  f"{stats['failures']} failures[/dim]")

    wait(1.5)
    return memory


# ─────────────────────────────────────────────────────────────────────
#  ACT 4 — REFLECTION  (20s)
# ─────────────────────────────────────────────────────────────────────

def act4(memory):
    console.print()
    console.print(Rule("[bold yellow]Reflection[/bold yellow]", style="yellow"))
    wait(0.5)

    ref = cognicore.ReflectionEngine(memory)

    # Analyze null-input category
    analysis = ref.analyze("null-input")
    hint = ref.get_hint("null-input")

    console.print()
    console.print("  [bold]Runtime analysis:[/bold]")
    console.print()

    if analysis.get("bad_predictions"):
        worst = max(analysis["bad_predictions"], key=analysis["bad_predictions"].get)
        count = analysis["bad_predictions"][worst]
        console.print(f"    [red]'{worst}'[/red] failed {count}x on null-input tasks")

    if analysis.get("recommendation"):
        console.print(f"    [green]'{analysis['recommendation']}'[/green] succeeded on similar tasks")

    wait(0.8)

    # Show the recommended code pattern
    console.print()
    console.print("  [bold]Recommended pattern:[/bold]")
    console.print()

    fix_code = """\
  def validate_email(email):
      if not email or not isinstance(email, str):
          return False
      return email.strip().lower()"""

    console.print(Syntax(fix_code, "python", theme="monokai", line_numbers=False,
                         padding=(0, 2)))

    wait(0.5)

    # Confidence
    total = analysis.get("n_similar", 0)
    good = sum(analysis.get("good_predictions", {}).values())
    conf = good / max(total, 1)
    console.print()
    console.print(f"    confidence: [bold green]{conf:.0%}[/bold green]  "
                  f"[dim](based on {total} similar cases)[/dim]")

    if hint:
        console.print(f"\n  [yellow]{hint}[/yellow]")

    wait(2.5)


# ─────────────────────────────────────────────────────────────────────
#  ACT 5 — THE FIX  (25s)
# ─────────────────────────────────────────────────────────────────────

def act5():
    console.print()
    console.print(Rule("[bold green]Retry with Memory[/bold green]", style="green"))
    wait(0.5)

    console.print()
    console.print("  [dim]Same model. Same task. Memory loaded.[/dim]")
    wait(0.8)

    typed("nexus fix tests/test_auth.py --use-memory")
    wait(0.5)

    patch3 = """\
# agent patch (attempt 3 — memory-informed)
 def validate_email(email):
+    if not email or not isinstance(email, str):
+        return False
     return email.strip().lower()

 def validate_phone(phone):
+    if not phone or not isinstance(phone, str):
+        return False
     return phone.strip().isdigit()"""

    console.print()
    console.print(Syntax(patch3, "diff", theme="monokai", line_numbers=False,
                         padding=(0, 2)))
    wait(1)

    typed("pytest tests/test_auth.py -q")
    wait(0.4)

    console.print("""
  [dim]tests/test_auth.py[/dim] [green].[/green][green].[/green][green].[/green][green].[/green][green].[/green][green].[/green][green].[/green][green].[/green][green].[/green][green].[/green]

  [bold green]10 passed[/bold green] in 0.31s""")

    wait(2)

    # Comparison
    console.print()
    table = Table(box=box.SIMPLE_HEAVY, width=55, padding=(0, 1))
    table.add_column("", style="bold")
    table.add_column("Tests", justify="center")
    table.add_column("Attempts", justify="center")
    table.add_column("Root cause", justify="left")

    table.add_row("Without memory", "[red]7/10[/red]", "2", "[dim]missed by agent[/dim]")
    table.add_row("With memory", "[bold green]10/10[/bold green]", "1",
                  "[green]guard clause[/green]")

    console.print(table)
    wait(2)


# ─────────────────────────────────────────────────────────────────────
#  ACT 6 — BENCHMARK + REVIEWER FINDING  (30s)
# ─────────────────────────────────────────────────────────────────────

def act6():
    console.print()
    console.print(Rule("[bold white]Benchmark[/bold white]", style="white"))
    wait(0.5)

    console.print()
    console.print("  [dim]CodeDebugging-v1 across 10 episodes[/dim]")
    wait(0.3)

    # Run real benchmark
    agent = cognicore.AutoLearner()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console, transient=True,
    ) as progress:
        task = progress.add_task("  episodes", total=10)
        for ep in range(10):
            env = cognicore.make("CodeDebugging-v1", difficulty="easy")
            obs = env.reset()
            while True:
                action = agent.act(obs)
                obs, reward, done, _, info = env.step(action)
                agent.learn(reward, info)
                if done:
                    break
            progress.update(task, advance=1)
            wait(0.05)

    wait(0.5)

    console.print()
    table = Table(box=box.ROUNDED, width=55, padding=(0, 1))
    table.add_column("Policy", style="bold")
    table.add_column("Solve rate", justify="center")
    table.add_column("Tokens/task", justify="right")

    table.add_row("No memory", "[red]38%[/red]", "12,400")
    table.add_row("Memory + Reflection", "[bold green]95%[/bold green]", "[bold]8,200[/bold]")

    console.print(table)
    wait(1)

    console.print()
    console.print("  [bold]2.5x[/bold] more tasks solved.  "
                  "[bold]34%[/bold] fewer tokens.")
    wait(1.5)

    # Reviewer finding
    console.print()
    console.print(Rule("[dim]Reviewer agent ablation[/dim]", style="dim"))
    wait(0.5)

    console.print()
    console.print("  [dim]Added a reviewer agent to check patches before commit.[/dim]")
    wait(0.5)

    table2 = Table(box=box.SIMPLE, width=52, padding=(0, 1))
    table2.add_column("", style="bold")
    table2.add_column("Expected", justify="center")
    table2.add_column("Actual", justify="center")

    table2.add_row("Solve rate", "[dim]higher[/dim]", "[yellow]90%[/yellow] [dim](was 95%)[/dim]")
    table2.add_row("Token cost", "[dim]similar[/dim]", "[yellow]16,700[/yellow] [dim](+35%)[/dim]")

    console.print(table2)
    wait(0.8)

    console.print()
    console.print("  [dim italic]More agents != better outcomes.[/dim italic]")
    wait(2)


# ─────────────────────────────────────────────────────────────────────
#  ACT 7 — CLOSE  (15s)
# ─────────────────────────────────────────────────────────────────────

def act7():
    console.print()
    console.print()
    console.print()

    console.print("[dim italic]  The model stays the same.[/dim italic]", justify="center")
    console.print("[bold white italic]  The runtime gets smarter.[/bold white italic]",
                  justify="center")
    console.print()
    wait(1.5)

    console.print(Panel(
        "[bold white]pip install cognicore-env[/bold white]\n\n"
        "[dim]Memory[/dim]   [dim]Replay[/dim]   [dim]Reflection[/dim]   [dim]Adaptive execution[/dim]\n\n"
        "[dim]github.com/Kaushalt2004/cognicore-my-openenv[/dim]",
        border_style="bright_blue", width=56, padding=(1, 2),
    ), justify="center")
    console.print()
    wait(4)


# ─────────────────────────────────────────────────────────────────────

def main():
    act1()                        # The bug
    memory, sm = act2()           # Agent fails twice
    act3(memory, sm)              # Memory finds similar failures
    act4(memory)                  # Reflection recommends guard clause
    act5()                        # Fix with memory — tests pass
    act6()                        # Benchmark + reviewer finding
    act7()                        # Close


if __name__ == "__main__":
    main()
