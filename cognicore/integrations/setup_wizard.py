"""
NEXUS Integration Setup Wizard — interactive CLI for configuring integrations.
"""
import os, sys, json
from pathlib import Path

CONFIG_DIR = Path.home() / ".cognicore"
CONFIG_FILE = CONFIG_DIR / "config.yml"
SCHEDULE_FILE = CONFIG_DIR / "schedule.yml"


def setup_wizard():
    """Interactive setup for NEXUS enterprise integrations."""
    CONFIG_DIR.mkdir(exist_ok=True)
    print("\n  ======================================")
    print("   NEXUS Integration Setup Wizard")
    print("  ======================================\n")

    config = {"nexus": {"version": "0.6.0", "local_port": 7842}, "integrations": {}, "defaults": {
        "policy": "test_first", "max_attempts": 3, "budget_per_task": 2.0}}

    # Step 1: Select integrations
    print("  Step 1: Which integrations do you want?\n")
    integrations = [
        ("github", "GitHub Issues (auto-fix labeled issues)"),
        ("ci", "CI Fixer (auto-repair failed workflows)"),
        ("slack", "Slack (mention @nexus to trigger)"),
        ("linear", "Linear (assign tickets to Nexus)"),
        ("scheduler", "Scheduled Tasks (nightly runs)"),
        ("pr_review", "PR Review (memory-backed reviews)"),
    ]
    selected = {}
    for key, desc in integrations:
        try:
            ans = input(f"    [{key}] {desc}? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = 'y'
        selected[key] = ans != 'n'

    # Step 2: GitHub setup
    if selected.get("github") or selected.get("ci") or selected.get("pr_review"):
        print("\n  Step 2: GitHub Setup\n")
        token = input("    GitHub token (or GITHUB_TOKEN env var): ").strip()
        if not token:
            token = os.environ.get("GITHUB_TOKEN", "")
        repo = input("    Repository to monitor (owner/repo): ").strip()
        secret = input("    Webhook secret (optional): ").strip()
        config["integrations"]["github"] = {
            "token": "$GITHUB_TOKEN" if not token else "****",
            "repo": repo, "webhook_secret": secret or "",
            "auto_pr": True, "label_trigger": "nexus",
            "label_solved": "nexus-solved", "label_failed": "nexus-failed"
        }
        if token:
            os.environ["GITHUB_TOKEN"] = token
        if repo:
            try:
                from cognicore.integrations.github_issues import GitHubIssuesTrigger
                gh = GitHubIssuesTrigger(config["integrations"]["github"])
                if gh.available:
                    issues = gh.list_nexus_issues(repo)
                    print(f"    Connected. Found {len(issues)} nexus-labeled issues.")
                else:
                    print("    Token set but PyGithub not installed: pip install PyGithub")
            except Exception as e:
                print(f"    Connection test: {e}")

    # Step 3: Slack setup
    if selected.get("slack"):
        print("\n  Step 3: Slack Setup\n")
        bot_token = input("    Bot token (or SLACK_BOT_TOKEN env var): ").strip()
        signing = input("    Signing secret: ").strip()
        config["integrations"]["slack"] = {
            "bot_token": "$SLACK_BOT_TOKEN",
            "signing_secret": signing or "",
            "update_interval": 10
        }
        if bot_token:
            os.environ["SLACK_BOT_TOKEN"] = bot_token

    # Step 4: Linear setup
    if selected.get("linear"):
        print("\n  Step 4: Linear Setup\n")
        api_key = input("    Linear API key: ").strip()
        config["integrations"]["linear"] = {"api_key": "$LINEAR_API_KEY", "trigger_label": "nexus"}
        if api_key:
            os.environ["LINEAR_API_KEY"] = api_key

    # Step 5: Scheduler
    if selected.get("scheduler"):
        print("\n  Step 5: Schedule Setup\n")
        try:
            nightly = input("    Run nightly bug fixes? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            nightly = 'y'
        if nightly != 'n':
            time_str = input("    What time? [02:00]: ").strip() or "02:00"
            h, m = time_str.split(":")
            schedule = {"schedules": [
                {"name": "nightly_bug_fix", "cron": f"{m} {h} * * *",
                 "task": "fix_labeled_issues", "repo": config.get("integrations", {}).get("github", {}).get("repo", "")}
            ]}
            # Write schedule
            with open(SCHEDULE_FILE, 'w') as f:
                try:
                    import yaml
                    yaml.dump(schedule, f)
                except ImportError:
                    json.dump(schedule, f, indent=2)
            print(f"    Scheduled: nightly at {time_str} UTC")
        config["integrations"]["scheduler"] = {"enabled": True, "timezone": "UTC"}

    # Step 6: Defaults
    print("\n  Step 6: Defaults\n")
    try:
        policy = input("    Default policy [test_first]: ").strip() or "test_first"
        budget = input("    Budget per task USD [2.00]: ").strip() or "2.00"
    except (EOFError, KeyboardInterrupt):
        policy, budget = "test_first", "2.00"
    config["defaults"]["policy"] = policy
    config["defaults"]["budget_per_task"] = float(budget)

    # Save config
    with open(CONFIG_FILE, 'w') as f:
        try:
            import yaml
            yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            json.dump(config, f, indent=2)
    print(f"\n  Config saved: {CONFIG_FILE}")

    # Summary
    active = [k for k, v in selected.items() if v]
    print(f"\n  ======================================")
    print(f"   Setup Complete!")
    print(f"  ======================================")
    print(f"  Active integrations: {', '.join(active)}")
    print(f"  Config: {CONFIG_FILE}")
    print(f"\n  Start NEXUS: cognicore ui")
    print(f"  Webhook URL: http://localhost:7842/webhooks/github")
    print()

    return config


def test_connections():
    """Test all configured integration connections."""
    print("\n  Testing integrations...\n")
    results = {}

    # GitHub
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        try:
            from github import Github
            g = Github(token)
            user = g.get_user().login
            results["github"] = f"Connected as {user}"
        except Exception as e:
            results["github"] = f"FAILED: {e}"
    else:
        results["github"] = "Not configured"

    # Slack
    slack_token = os.environ.get("SLACK_BOT_TOKEN", "")
    if slack_token:
        try:
            from slack_sdk import WebClient
            c = WebClient(token=slack_token)
            info = c.auth_test()
            results["slack"] = f"Connected as {info['bot_id']}"
        except Exception as e:
            results["slack"] = f"FAILED: {e}"
    else:
        results["slack"] = "Not configured"

    # Task queue
    try:
        from cognicore.integrations.task_queue import NexusTaskQueue
        q = NexusTaskQueue()
        s = q.stats()
        results["task_queue"] = f"OK ({s['total']} tasks)"
    except Exception as e:
        results["task_queue"] = f"FAILED: {e}"

    for name, status in results.items():
        icon = "OK" if "Connected" in status or "OK" in status else "!!"
        print(f"    [{icon}] {name}: {status}")
    print()
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_connections()
    else:
        setup_wizard()
