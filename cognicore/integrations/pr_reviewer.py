"""
PR Auto-Reviewer — uses CogniCore memory to review PRs.
Posts memory-informed review comments on risky patterns.
"""
import os, json, re
from cognicore.integrations.task_queue import NexusTask, TaskSource
from cognicore.research.persistent_store import PersistentCognitionStore

try:
    from github import Github
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False


class PRReviewer:
    """Memory-backed PR reviewer using CogniCore persistent cognition."""

    def __init__(self, config=None):
        self.config = config or {}
        self.token = self.config.get("token", os.environ.get("GITHUB_TOKEN", ""))
        self.min_confidence = self.config.get("min_confidence", 0.7)
        self.monitored_repos = self.config.get("repos", [])
        self._gh = Github(self.token) if self.token and GITHUB_AVAILABLE else None
        self._memory = PersistentCognitionStore()

    def create_task(self, webhook_payload: dict) -> NexusTask:
        pr = webhook_payload.get("pull_request", {})
        repo = webhook_payload.get("repository", {})
        return NexusTask(
            source=TaskSource.PR_REVIEW.value,
            repo=repo.get("full_name", ""),
            title=f"Review PR #{pr.get('number', '?')}: {pr.get('title', '')}",
            description=pr.get("body", "") or pr.get("title", ""),
            priority=2,
            policy="minimal",
            metadata={
                "pr_number": pr.get("number"),
                "pr_url": pr.get("html_url", ""),
                "repo_full_name": repo.get("full_name", ""),
                "head_sha": pr.get("head", {}).get("sha", ""),
                "author": pr.get("user", {}).get("login", ""),
                "diff_url": pr.get("diff_url", ""),
            }
        )

    def review_pr(self, repo_name: str, pr_number: int) -> dict:
        """Review a PR using CogniCore memory."""
        if not self._gh:
            return {"error": "GitHub not configured"}

        try:
            repo = self._gh.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
            files = pr.get_files()

            findings = []
            total_risk = 0

            for f in files:
                if not f.filename.endswith('.py'):
                    continue
                patch = f.patch or ""
                file_findings = self._analyze_patch(f.filename, patch)
                findings.extend(file_findings)
                total_risk += sum(f.get("risk", 0) for f in file_findings)

            risk_score = min(10, total_risk)
            risk_label = "Low" if risk_score < 4 else "Medium" if risk_score < 7 else "High"

            return {
                "findings": findings,
                "risk_score": risk_score,
                "risk_label": risk_label,
                "files_reviewed": len(list(pr.get_files())),
            }
        except Exception as e:
            return {"error": str(e)}

    def _analyze_patch(self, filename: str, patch: str) -> list:
        """Analyze a file patch against CogniCore memory."""
        findings = []

        # Pattern detectors
        patterns = [
            {"regex": r"(\w+)\[(['\"]?\w+['\"]?)\]",
             "category": "none_handling",
             "msg": "Dict/list access without None check",
             "suggestion": "Use .get() with default value"},
            {"regex": r"len\(\w+\)\s*[<>]=?\s*\d+.*\[",
             "category": "off_by_one",
             "msg": "Potential off-by-one in boundary check",
             "suggestion": "Verify boundary includes edge case"},
            {"regex": r"except\s*:",
             "category": "error_handling",
             "msg": "Bare except clause catches all exceptions",
             "suggestion": "Specify exception type"},
            {"regex": r"==\s*None|!=\s*None",
             "category": "none_handling",
             "msg": "Use 'is None' instead of '== None'",
             "suggestion": "Replace with 'is None' or 'is not None'"},
            {"regex": r"os\.system\(|subprocess\.call\(",
             "category": "security",
             "msg": "Shell command execution detected",
             "suggestion": "Use subprocess.run with shell=False"},
            {"regex": r"\.format\(|%\s*\(",
             "category": "security",
             "msg": "String formatting in potential SQL/command context",
             "suggestion": "Use parameterized queries"},
        ]

        lines = patch.split('\n')
        for i, line in enumerate(lines):
            if not line.startswith('+') or line.startswith('+++'):
                continue

            for pat in patterns:
                if re.search(pat["regex"], line):
                    # Check memory for this pattern
                    insights = self._memory.get_cross_session_insights(pat["category"])
                    failed = insights.get("failed_tactics", {})
                    succeeded = insights.get("successful_tactics", {})
                    total_memory = sum(failed.values()) + sum(succeeded.values())

                    if total_memory > 0:
                        fail_rate = sum(failed.values()) / total_memory
                        confidence = min(1.0, total_memory / 10)

                        if confidence >= self.min_confidence:
                            findings.append({
                                "file": filename, "line": i + 1,
                                "pattern": pat["category"],
                                "msg": pat["msg"],
                                "suggestion": pat["suggestion"],
                                "memory_matches": total_memory,
                                "fail_rate": round(fail_rate * 100),
                                "confidence": round(confidence * 100),
                                "risk": 2 if fail_rate > 0.5 else 1,
                            })
        return findings

    def post_review(self, repo_name: str, pr_number: int):
        """Post a memory-informed review on a PR."""
        if not self._gh:
            return

        result = self.review_pr(repo_name, pr_number)
        if result.get("error"):
            return

        findings = result["findings"]
        if not findings:
            return  # Nothing to report

        # Build review body
        body = f"## NEXUS Memory Review\n\n"
        body += f"Found **{len(findings)}** patterns matching past failures:\n\n"

        for f in findings:
            icon = "warning" if f["risk"] > 1 else "information_source"
            body += (f":{icon}: **{f['file']}** L{f['line']}: {f['msg']}\n"
                    f"  Memory: This pattern has {f['memory_matches']} "
                    f"historical matches ({f['fail_rate']}% failure rate)\n"
                    f"  Suggestion: {f['suggestion']}\n\n")

        body += (f"\n{'checkmark' if result['risk_score'] < 4 else 'warning'} "
                f"**Overall risk: {result['risk_label']} ({result['risk_score']}/10)**\n"
                f"Based on CogniCore persistent memory\n\n"
                f"*Automated by [NEXUS](https://github.com/Kaushalt2004/cognicore-my-openenv)*")

        try:
            repo = self._gh.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
            pr.create_issue_comment(body)
        except Exception as e:
            print(f"  [PR Review] Failed to post: {e}")
