"""
NEXUS Webhook Server — receives events from GitHub, Slack, Linear.
Unified ingestion point for all enterprise integrations.
"""
import json, hmac, hashlib
from fastapi import APIRouter, Request, HTTPException, Header
from typing import Optional

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Will be set by server.py on startup
_queue = None
_config = None


def init(queue, config):
    global _queue, _config
    _queue = queue
    _config = config


@router.post("/github")
async def github_webhook(request: Request,
                         x_hub_signature_256: Optional[str] = Header(None),
                         x_github_event: Optional[str] = Header(None)):
    """Handle GitHub webhooks (issues, CI, PRs)."""
    body = await request.body()

    # Verify signature if secret configured
    secret = (_config or {}).get("github", {}).get("webhook_secret", "")
    if secret and x_hub_signature_256:
        expected = "sha256=" + hmac.new(
            secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, x_hub_signature_256):
            raise HTTPException(403, "Invalid signature")

    payload = json.loads(body)
    event = x_github_event or "unknown"

    if event == "issues":
        return _handle_issue(payload)
    elif event == "check_run" or event == "workflow_run":
        return _handle_ci(payload)
    elif event == "pull_request":
        return _handle_pr(payload)

    return {"ok": True, "event": event, "action": "ignored"}


def _handle_issue(payload):
    from cognicore.integrations.github_issues import GitHubIssuesTrigger
    action = payload.get("action", "")
    issue = payload.get("issue", {})
    labels = [l["name"] for l in issue.get("labels", [])]
    trigger_label = (_config or {}).get("github", {}).get("label_trigger", "nexus")

    if action == "labeled" and trigger_label in labels:
        trigger = GitHubIssuesTrigger(_config.get("github", {}))
        task = trigger.create_task(payload)
        if _queue:
            task_id = _queue.submit(task)
            return {"ok": True, "task_id": task_id, "source": "github_issue"}
    return {"ok": True, "action": "skipped"}


def _handle_ci(payload):
    from cognicore.integrations.ci_fixer import CIFailureFixer
    status = payload.get("workflow_run", {}).get("conclusion", "")
    if status == "failure":
        fixer = CIFailureFixer(_config.get("ci", {}))
        task = fixer.create_task(payload)
        if _queue:
            task_id = _queue.submit(task)
            return {"ok": True, "task_id": task_id, "source": "ci_failure"}
    return {"ok": True, "action": "skipped"}


def _handle_pr(payload):
    from cognicore.integrations.pr_reviewer import PRReviewer
    action = payload.get("action", "")
    if action in ("opened", "synchronize"):
        reviewer = PRReviewer(_config.get("pr_review", {}))
        task = reviewer.create_task(payload)
        if _queue:
            task_id = _queue.submit(task)
            return {"ok": True, "task_id": task_id, "source": "pr_review"}
    return {"ok": True, "action": "skipped"}


@router.post("/slack/events")
async def slack_webhook(request: Request):
    """Handle Slack events (mentions, slash commands)."""
    body = await request.json()

    # URL verification challenge
    if body.get("type") == "url_verification":
        return {"challenge": body["challenge"]}

    event = body.get("event", {})
    event_type = event.get("type", "")

    if event_type == "app_mention":
        from cognicore.integrations.slack import SlackIntegration
        slack = SlackIntegration(_config.get("slack", {}))
        task = slack.create_task(event)
        if _queue:
            task_id = _queue.submit(task)
            slack.reply_started(event, task_id)
            return {"ok": True, "task_id": task_id}

    return {"ok": True}


@router.post("/slack/commands")
async def slack_command(request: Request):
    """Handle Slack slash commands."""
    form = await request.form()
    command = form.get("command", "")
    text = form.get("text", "").strip()

    if command == "/nexus":
        from cognicore.integrations.slack import SlackIntegration
        slack = SlackIntegration(_config.get("slack", {}))
        return slack.handle_command(text, dict(form))

    return {"text": "Unknown command"}


@router.post("/linear")
async def linear_webhook(request: Request):
    """Handle Linear webhooks."""
    body = await request.json()
    action = body.get("action", "")
    data = body.get("data", {})

    if action == "update":
        labels = [l["name"] for l in data.get("labels", [])]
        assignee = data.get("assignee", {}).get("name", "")
        trigger_label = (_config or {}).get("linear", {}).get("trigger_label", "nexus")

        if trigger_label in labels or assignee.lower() == "nexus":
            from cognicore.integrations.linear import LinearIntegration
            linear = LinearIntegration(_config.get("linear", {}))
            task = linear.create_task(data)
            if _queue:
                task_id = _queue.submit(task)
                linear.update_status(data["id"], "In Progress")
                return {"ok": True, "task_id": task_id}

    return {"ok": True}


@router.get("/health")
async def health():
    """Integration health check."""
    status = {}
    cfg = _config or {}
    status["github"] = bool(cfg.get("github", {}).get("token"))
    status["slack"] = bool(cfg.get("slack", {}).get("bot_token"))
    status["linear"] = bool(cfg.get("linear", {}).get("api_key"))
    status["scheduler"] = cfg.get("scheduler", {}).get("enabled", False)
    status["queue"] = _queue.stats() if _queue else {}
    return status
