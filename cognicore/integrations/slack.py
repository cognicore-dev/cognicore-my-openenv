"""
Slack Integration — @nexus mention triggers tasks, slash commands for status.
Live thread updates as NEXUS processes tasks.
"""
import os, json, time
from cognicore.integrations.task_queue import NexusTask, TaskSource

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    class WebClient:
        def __init__(self, *a, **kw): pass


class SlackIntegration:
    """Slack bot that triggers NEXUS tasks and streams updates."""

    def __init__(self, config=None):
        self.config = config or {}
        self.token = self.config.get("bot_token", os.environ.get("SLACK_BOT_TOKEN", ""))
        self.update_interval = self.config.get("update_interval", 10)
        self.client = WebClient(token=self.token) if self.token and SLACK_AVAILABLE else None

    @property
    def available(self):
        return self.client is not None and bool(self.token)

    def create_task(self, event: dict) -> NexusTask:
        """Create NexusTask from Slack mention event."""
        text = event.get("text", "")
        # Remove bot mention
        import re
        clean = re.sub(r'<@[A-Z0-9]+>', '', text).strip()

        return NexusTask(
            source=TaskSource.SLACK.value,
            title=clean[:100] or "Slack task",
            description=clean,
            priority=2,
            policy="test_first",
            metadata={
                "channel": event.get("channel", ""),
                "thread_ts": event.get("thread_ts", event.get("ts", "")),
                "user": event.get("user", ""),
                "ts": event.get("ts", ""),
            }
        )

    def reply_started(self, event: dict, task_id: str):
        """Reply that NEXUS is working on it."""
        if not self.available:
            return
        channel = event.get("channel", "")
        thread = event.get("thread_ts", event.get("ts", ""))
        try:
            self.client.chat_postMessage(
                channel=channel, thread_ts=thread,
                text=f":arrows_counterclockwise: NEXUS is working on this... (task `{task_id}`)",
                unfurl_links=False)
        except Exception:
            pass

    def post_update(self, channel: str, thread_ts: str, step: str, status: str):
        """Post a live update to the Slack thread."""
        if not self.available:
            return
        icon = {"pass": ":white_check_mark:", "fail": ":x:",
                "running": ":arrows_counterclockwise:"}.get(status, ":grey_question:")
        try:
            self.client.chat_postMessage(
                channel=channel, thread_ts=thread_ts,
                text=f"{icon} {step}", unfurl_links=False)
        except Exception:
            pass

    def post_result(self, task: NexusTask):
        """Post final result to Slack thread."""
        if not self.available:
            return
        meta = task.metadata
        channel = meta.get("channel", "")
        thread = meta.get("thread_ts", "")
        if not channel:
            return

        result = task.result
        if task.status == "solved":
            blocks = [
                {"type": "section", "text": {"type": "mrkdwn", "text":
                    f":white_check_mark: *NEXUS solved this!*\n\n"
                    f"*Attempts:* {result.get('attempts', '?')}\n"
                    f"*Tokens:* {result.get('tokens', 0):,}\n"
                    f"*Cost:* ${result.get('cost', 0):.4f}\n"
                    f"*Policy:* `{result.get('policy', task.policy)}`"
                }},
            ]
            pr = result.get("pr_url")
            if pr:
                blocks.append({"type": "section", "text": {"type": "mrkdwn",
                    "text": f":link: <{pr}|View Pull Request>"}})
        else:
            blocks = [
                {"type": "section", "text": {"type": "mrkdwn", "text":
                    f":x: *NEXUS could not solve this*\n\n"
                    f"*Attempts:* {result.get('attempts', '?')}\n"
                    f"*Suggestion:* {result.get('reflection', 'Manual review needed')}"
                }},
            ]

        try:
            self.client.chat_postMessage(
                channel=channel, thread_ts=thread,
                blocks=blocks, text="NEXUS result")
        except Exception:
            pass

    def handle_command(self, text: str, form_data: dict) -> dict:
        """Handle /nexus slash commands."""
        parts = text.split(None, 1)
        cmd = parts[0] if parts else "help"
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "fix":
            task = NexusTask(
                source=TaskSource.SLACK.value,
                title=arg[:100] or "Slack fix request",
                description=arg,
                metadata={"channel": form_data.get("channel_id", ""),
                          "user": form_data.get("user_id", "")})
            return {"response_type": "in_channel",
                    "text": f":arrows_counterclockwise: Task submitted: `{task.id}`"}

        elif cmd == "status":
            from cognicore.integrations.task_queue import NexusTaskQueue
            q = NexusTaskQueue()
            stats = q.stats()
            return {"text": (
                f"*NEXUS Status*\n"
                f"Total tasks: {stats['total']}\n"
                f"By status: {json.dumps(stats['by_status'])}\n"
                f"Dead letter: {stats['dead_letter']}")}

        elif cmd == "cost":
            from cognicore.nexus.trajectory_store import TrajectoryStore
            ts = TrajectoryStore()
            s = ts.get_stats()
            return {"text": f"*Token Usage*\nTrajectories: {s['total_trajectories']}\n"
                           f"Solved: {s['solved']}"}

        return {"text": (
            "*NEXUS Commands*\n"
            "`/nexus fix <description>` - Submit a fix task\n"
            "`/nexus status` - Show queue status\n"
            "`/nexus cost` - Show token usage\n"
            "`/nexus memory <repo>` - Show memory stats")}
