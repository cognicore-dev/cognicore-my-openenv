"""
GitHub Issues Integration — auto-trigger NEXUS on labeled issues.
Labels issue with 'nexus' -> NEXUS fixes -> opens PR -> comments result.
"""
import os, json
from datetime import datetime
from cognicore.integrations.task_queue import NexusTask, TaskSource

try:
    from github import Github, GithubException
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False


class GitHubIssuesTrigger:
    """Monitors GitHub issues and triggers NEXUS tasks."""

    def __init__(self, config=None):
        self.config = config or {}
        self.token = self.config.get("token", os.environ.get("GITHUB_TOKEN", ""))
        self.auto_pr = self.config.get("auto_pr", True)
        self.label_trigger = self.config.get("label_trigger", "nexus")
        self.label_solved = self.config.get("label_solved", "nexus-solved")
        self.label_failed = self.config.get("label_failed", "nexus-failed")
        self._gh = Github(self.token) if self.token and GITHUB_AVAILABLE else None

    @property
    def available(self):
        return self._gh is not None

    def create_task(self, webhook_payload: dict) -> NexusTask:
        """Create a NexusTask from a GitHub issue webhook payload."""
        issue = webhook_payload.get("issue", {})
        repo = webhook_payload.get("repository", {})

        return NexusTask(
            source=TaskSource.GITHUB_ISSUE.value,
            repo=repo.get("full_name", ""),
            title=issue.get("title", ""),
            description=issue.get("body", "") or issue.get("title", ""),
            priority=1 if "critical" in [l["name"] for l in issue.get("labels", [])] else 2,
            policy=self.config.get("default_policy", "test_first"),
            max_attempts=self.config.get("max_attempts", 3),
            budget_usd=self.config.get("budget_per_task", 2.0),
            callback_url=issue.get("url", ""),
            metadata={
                "issue_number": issue.get("number"),
                "issue_url": issue.get("html_url", ""),
                "repo_full_name": repo.get("full_name", ""),
                "author": issue.get("user", {}).get("login", ""),
                "labels": [l["name"] for l in issue.get("labels", [])],
            }
        )

    def on_task_completed(self, task: NexusTask):
        """Called when NEXUS finishes processing a task."""
        if not self._gh:
            return

        meta = task.metadata
        repo_name = meta.get("repo_full_name", task.repo)
        issue_num = meta.get("issue_number")
        if not repo_name or not issue_num:
            return

        try:
            repo = self._gh.get_repo(repo_name)
            issue = repo.get_issue(issue_num)
            result = task.result

            if task.status == "solved":
                # Post success comment
                pr_url = result.get("pr_url", "N/A")
                comment = (
                    f"## NEXUS Solved This\n\n"
                    f"**Solved in {result.get('attempts', '?')} attempt(s)**\n\n"
                    f"| Metric | Value |\n|---|---|\n"
                    f"| Tokens | {result.get('tokens', 0):,} |\n"
                    f"| Cost | ${result.get('cost', 0):.4f} |\n"
                    f"| Policy | `{result.get('policy', task.policy)}` |\n"
                    f"| Memory hits | {result.get('memory_hits', 0)} |\n"
                    f"| PR | {pr_url} |\n\n"
                    f"*Automated by [NEXUS](https://github.com/Kaushalt2004/cognicore-my-openenv)*"
                )
                issue.create_comment(comment)

                # Update labels
                try:
                    issue.remove_from_labels(self.label_trigger)
                except GithubException:
                    pass
                issue.add_to_labels(self.label_solved)

                # Open PR if auto_pr enabled
                if self.auto_pr and result.get("patch"):
                    self._open_pr(repo, issue, result)

            else:
                # Post failure comment
                comment = (
                    f"## NEXUS Could Not Solve This\n\n"
                    f"**Attempted {result.get('attempts', '?')} time(s)**\n\n"
                    f"| Metric | Value |\n|---|---|\n"
                    f"| Tokens | {result.get('tokens', 0):,} |\n"
                    f"| Categories tried | {result.get('tactics', 'unknown')} |\n\n"
                    f"**Suggestion**: {result.get('reflection', 'Manual review needed')}\n\n"
                    f"*Automated by [NEXUS](https://github.com/Kaushalt2004/cognicore-my-openenv)*"
                )
                issue.create_comment(comment)

                try:
                    issue.remove_from_labels(self.label_trigger)
                except GithubException:
                    pass
                issue.add_to_labels(self.label_failed)

        except Exception as e:
            print(f"  [GitHub] Error posting result: {e}")

    def _open_pr(self, repo, issue, result):
        """Open a PR with the fix."""
        try:
            branch_name = f"nexus/fix-{issue.number}"
            base = repo.default_branch
            ref = repo.get_git_ref(f"heads/{base}")

            # Create branch
            try:
                repo.create_git_ref(f"refs/heads/{branch_name}", ref.object.sha)
            except GithubException:
                pass  # branch exists

            # Create PR
            pr = repo.create_pull(
                title=f"fix: {issue.title} (NEXUS #{issue.number})",
                body=(f"Fixes #{issue.number}\n\n"
                      f"Automated fix by NEXUS.\n"
                      f"Policy: `{result.get('policy', 'test_first')}`\n"
                      f"Tokens: {result.get('tokens', 0):,}"),
                head=branch_name, base=base)
            return pr.html_url
        except Exception as e:
            print(f"  [GitHub] PR creation failed: {e}")
            return None

    def list_nexus_issues(self, repo_name: str) -> list:
        """List all issues with the nexus label."""
        if not self._gh:
            return []
        try:
            repo = self._gh.get_repo(repo_name)
            issues = repo.get_issues(labels=[self.label_trigger], state="open")
            return [{"number": i.number, "title": i.title, "url": i.html_url}
                    for i in issues]
        except Exception:
            return []
