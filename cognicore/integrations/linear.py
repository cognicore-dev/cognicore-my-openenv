"""
Linear Integration — trigger NEXUS from Linear tickets.
Assign to "Nexus" or label with "nexus" to auto-trigger.
"""
import os, json
from cognicore.integrations.task_queue import NexusTask, TaskSource

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class LinearIntegration:
    """Linear ticket management integration."""

    API_URL = "https://api.linear.app/graphql"

    def __init__(self, config=None):
        self.config = config or {}
        self.api_key = self.config.get("api_key", os.environ.get("LINEAR_API_KEY", ""))
        self.trigger_label = self.config.get("trigger_label", "nexus")

    @property
    def available(self):
        return bool(self.api_key) and HTTPX_AVAILABLE

    def create_task(self, issue_data: dict) -> NexusTask:
        priority_map = {0: 3, 1: 0, 2: 1, 3: 2, 4: 3}
        return NexusTask(
            source=TaskSource.LINEAR.value,
            title=issue_data.get("title", ""),
            description=issue_data.get("description", "") or issue_data.get("title", ""),
            priority=priority_map.get(issue_data.get("priority", 3), 2),
            policy="test_first",
            metadata={
                "linear_id": issue_data.get("id", ""),
                "identifier": issue_data.get("identifier", ""),
                "team": issue_data.get("team", {}).get("name", ""),
                "url": issue_data.get("url", ""),
            }
        )

    def update_status(self, issue_id: str, status_name: str):
        if not self.available:
            return
        # Map status names to Linear state IDs (would need actual team config)
        mutation = """mutation { issueUpdate(id: "%s", input: { stateId: "%s" }) { success } }"""
        # In production, resolve stateId from status_name via API
        print(f"  [Linear] Status update: {issue_id} -> {status_name}")

    def post_comment(self, issue_id: str, body: str):
        if not self.available:
            return
        query = """mutation($id: String!, $body: String!) {
            commentCreate(input: { issueId: $id, body: $body }) { success }
        }"""
        try:
            httpx.post(self.API_URL, json={"query": query,
                "variables": {"id": issue_id, "body": body}},
                headers={"Authorization": self.api_key, "Content-Type": "application/json"})
        except Exception as e:
            print(f"  [Linear] Comment failed: {e}")

    def on_task_completed(self, task: NexusTask):
        meta = task.metadata
        issue_id = meta.get("linear_id", "")
        if not issue_id:
            return

        result = task.result
        if task.status == "solved":
            self.update_status(issue_id, "Done")
            self.post_comment(issue_id,
                f"NEXUS solved this in {result.get('attempts', '?')} attempts.\n"
                f"Tokens: {result.get('tokens', 0):,} | Cost: ${result.get('cost', 0):.4f}\n"
                f"PR: {result.get('pr_url', 'N/A')}")
        else:
            self.update_status(issue_id, "Blocked")
            self.post_comment(issue_id,
                f"NEXUS attempted {result.get('attempts', '?')} times without success.\n"
                f"Suggestion: {result.get('reflection', 'Manual review needed')}")
