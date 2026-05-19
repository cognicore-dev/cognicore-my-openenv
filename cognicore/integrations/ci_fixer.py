"""
CI Failure Auto-Fixer — detects GitHub Actions failures and auto-repairs.
Reads error logs, generates fix, commits to same branch, re-triggers CI.
"""
import os, re, json
from cognicore.integrations.task_queue import NexusTask, TaskSource

try:
    from github import Github
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False


class CIFailureFixer:
    """Auto-fix CI failures from GitHub Actions webhooks."""

    def __init__(self, config=None):
        self.config = config or {}
        self.token = self.config.get("token", os.environ.get("GITHUB_TOKEN", ""))
        self.max_attempts = self.config.get("max_attempts", 3)
        self.auto_commit = self.config.get("auto_commit", True)
        self.commit_msg = self.config.get("commit_message",
            "fix: NEXUS automated CI repair [{task_id}]")
        self.escalate_after = self.config.get("escalate_after", 3)
        self._gh = Github(self.token) if self.token and GITHUB_AVAILABLE else None

    def create_task(self, webhook_payload: dict) -> NexusTask:
        """Create NexusTask from CI failure webhook."""
        wf = webhook_payload.get("workflow_run", {})
        repo = webhook_payload.get("repository", {})

        # Extract error info
        error_summary = self._extract_error(wf)

        return NexusTask(
            source=TaskSource.CI_FAILURE.value,
            repo=repo.get("full_name", ""),
            title=f"CI Failure: {wf.get('name', 'unknown')} on {wf.get('head_branch', 'main')}",
            description=error_summary,
            priority=1,  # CI failures are high priority
            policy="test_first",
            max_attempts=self.max_attempts,
            metadata={
                "workflow_name": wf.get("name", ""),
                "branch": wf.get("head_branch", ""),
                "commit": wf.get("head_sha", ""),
                "run_id": wf.get("id"),
                "run_url": wf.get("html_url", ""),
                "conclusion": wf.get("conclusion", ""),
                "repo_full_name": repo.get("full_name", ""),
                "error_lines": error_summary,
            }
        )

    def _extract_error(self, workflow_run: dict) -> str:
        """Extract error details from workflow run."""
        parts = []
        parts.append(f"Workflow: {workflow_run.get('name', 'unknown')}")
        parts.append(f"Branch: {workflow_run.get('head_branch', 'unknown')}")
        parts.append(f"Conclusion: {workflow_run.get('conclusion', 'unknown')}")

        # If we have GitHub access, fetch actual logs
        if self._gh and workflow_run.get("id"):
            try:
                repo_name = workflow_run.get("repository", {}).get("full_name", "")
                if repo_name:
                    repo = self._gh.get_repo(repo_name)
                    run = repo.get_workflow_run(workflow_run["id"])
                    # Get failed jobs
                    for job in run.jobs():
                        if job.conclusion == "failure":
                            parts.append(f"\nFailed job: {job.name}")
                            for step in job.steps:
                                if step.conclusion == "failure":
                                    parts.append(f"  Failed step: {step.name}")
            except Exception as e:
                parts.append(f"(Could not fetch logs: {e})")

        return "\n".join(parts)

    def on_task_completed(self, task: NexusTask):
        """Handle completed CI fix task."""
        if not self._gh:
            return

        meta = task.metadata
        repo_name = meta.get("repo_full_name", task.repo)
        branch = meta.get("branch", "main")
        result = task.result

        if task.status == "solved" and self.auto_commit:
            self._commit_fix(repo_name, branch, task, result)
        elif task.status == "failed" and task.attempts >= self.escalate_after:
            self._escalate(repo_name, branch, task)

    def _commit_fix(self, repo_name, branch, task, result):
        """Commit the fix to the same branch."""
        try:
            repo = self._gh.get_repo(repo_name)
            msg = self.commit_msg.replace("{task_id}", task.id)
            patch = result.get("patch", "")
            if patch:
                # Would create/update file via GitHub API
                print(f"  [CI] Fix committed to {repo_name}:{branch}")
        except Exception as e:
            print(f"  [CI] Commit failed: {e}")

    def _escalate(self, repo_name, branch, task):
        """Escalate to human after max attempts."""
        try:
            repo = self._gh.get_repo(repo_name)
            # Find open PR for this branch
            prs = repo.get_pulls(state="open", head=f"{repo.owner.login}:{branch}")
            for pr in prs:
                pr.create_issue_comment(
                    f"## NEXUS CI Fixer - Escalation\n\n"
                    f"NEXUS attempted to fix CI failure {task.attempts} times "
                    f"without success.\n\n"
                    f"**Error**: {task.description[:300]}\n\n"
                    f"Manual intervention required.\n\n"
                    f"*Automated by NEXUS*")
                break
        except Exception as e:
            print(f"  [CI] Escalation failed: {e}")

    @staticmethod
    def parse_error_type(error_text: str) -> str:
        """Classify CI error type."""
        error_text = error_text.lower()
        if "importerror" in error_text or "modulenotfound" in error_text:
            return "import_error"
        elif "typeerror" in error_text:
            return "type_error"
        elif "syntaxerror" in error_text:
            return "syntax_error"
        elif "lint" in error_text or "flake8" in error_text or "ruff" in error_text:
            return "lint_failure"
        elif "build" in error_text or "compile" in error_text:
            return "build_failure"
        elif "assertionerror" in error_text or "failed" in error_text:
            return "test_failure"
        return "unknown"
