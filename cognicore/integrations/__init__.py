"""
CogniCore Enterprise Integrations — NEXUS runs autonomously inside engineering teams.

Integrations:
  - GitHub Issues: auto-fix labeled issues, open PRs
  - CI Fixer: auto-repair failed GitHub Actions
  - Slack: @nexus mention triggers, slash commands
  - Linear: ticket assignment triggers
  - Scheduler: cron-based recurring tasks
  - PR Reviewer: memory-backed code review

Usage:
  cognicore integrations setup    # Interactive setup
  cognicore integrations test     # Test connections
  cognicore integrations status   # Health check
  cognicore webhooks start        # Start webhook server
"""
from cognicore.integrations.task_queue import NexusTask, NexusTaskQueue, TaskSource, TaskPriority

__all__ = [
    "NexusTask", "NexusTaskQueue", "TaskSource", "TaskPriority",
]
