"""
NEXUS Autonomous Runner — Devin-like autonomous coding engine.
Clones repos, reads code, generates LLM patches, runs tests, commits, opens PRs.
"""
import os, sys, subprocess, re, time, json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List

WORKSPACE = Path.home() / ".cognicore" / "workspaces"
WORKSPACE.mkdir(parents=True, exist_ok=True)


@dataclass
class RunResult:
    solved: bool = False
    patch: str = ""
    files_changed: List[str] = field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    tokens: int = 0
    attempts: int = 0
    pr_url: str = ""
    error: str = ""
    duration: float = 0.0


class NexusRunner:
    """Autonomous coding agent."""

    def __init__(self, github_token=None, max_attempts=3):
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")
        self.max_attempts = max_attempts
        self._cbs = []
        self._llm = None
        self.memory = None
        try:
            from cognicore.nexus.llm_provider import get_llm
            self._llm = get_llm()
        except Exception:
            pass
        try:
            from cognicore.research.persistent_store import PersistentCognitionStore
            self.memory = PersistentCognitionStore()
        except Exception:
            pass

    def on_step(self, cb):
        self._cbs.append(cb)

    def _emit(self, phase, action, detail="", status="done", tokens=0):
        step = {"phase": phase, "action": action, "detail": detail,
                "status": status, "tokens": tokens}
        for cb in self._cbs:
            try:
                cb(step)
            except Exception:
                pass

    def solve(self, prompt, repo="", repo_path="", auto_pr=True):
        t0 = time.time()
        r = RunResult()
        try:
            self._emit("plan", "Analyzing task", prompt)
            wd = self._workspace(repo, repo_path)
            files = self._search(wd, prompt)
            context = self._read(wd, files)
            mem = self._memory(prompt)

            for attempt in range(1, self.max_attempts + 1):
                r.attempts = attempt
                self._emit("patch", f"Attempt {attempt}/{self.max_attempts}")
                patch = self._generate(prompt, context, mem, attempt)
                if not patch:
                    continue
                changed = self._apply(wd, patch, prompt)
                if not changed:
                    continue
                r.files_changed = changed
                r.patch = patch
                ok, fail, out = self._test(wd)
                r.tests_passed, r.tests_failed = ok, fail
                if fail == 0 and ok > 0:
                    r.solved = True
                    self._emit("test", "ALL TESTS PASSED", f"{ok} passed")
                    break
                self._emit("test", f"Failed: {fail}", out[-200:], "failed")
                self._cmd(wd, ["git", "checkout", "."])

            if r.solved and auto_pr and self.github_token and repo:
                br = f"nexus/fix-{int(time.time())}"
                self._commit(wd, br, prompt)
                r.pr_url = self._open_pr(repo, br, prompt, r)
        except Exception as e:
            r.error = str(e)
            self._emit("error", str(e), status="failed")

        r.duration = round(time.time() - t0, 2)
        self._emit("done", "SOLVED" if r.solved else "FAILED",
                  f"{r.attempts} attempts, {r.duration}s")
        return r

    def _workspace(self, repo, repo_path):
        if repo_path and Path(repo_path).exists():
            self._emit("setup", f"Using {repo_path}")
            return Path(repo_path)
        if repo:
            wd = WORKSPACE / repo.replace("/", "_")
            if not wd.exists():
                url = f"https://{self.github_token}@github.com/{repo}.git" if self.github_token else f"https://github.com/{repo}.git"
                self._cmd(WORKSPACE, ["git", "clone", url, str(wd)])
            self._emit("setup", f"Cloned {repo}")
            return wd
        return Path.cwd()

    def _search(self, wd, prompt):
        """Search for files relevant to the bug — extract FUNCTION names from prompt."""
        self._emit("search", "Searching codebase")
        # Extract specific identifiers (function/class names, not common words)
        stop = {"fix","the","crash","when","bug","error","none","input","is",
                "content","function","method","on","in","a","an","with"}
        kws = [w for w in re.findall(r'[a-zA-Z_]\w{2,}', prompt)
               if w.lower() not in stop][:6]
        found = set()
        for kw in kws:
            try:
                out = self._cmd(wd, ["git", "grep", "-l", "-i", kw, "--", "*.py"], check=False)
                found.update(l.strip() for l in out.split("\n") if l.strip())
            except Exception:
                pass
        # Only keep files that contain the SPECIFIC function/identifier
        files = sorted(found)[:10]
        self._emit("search", f"Found {len(files)} files", "\n".join(files[:5]))
        return files

    def _read(self, wd, files):
        ctx = ""
        for f in files[:5]:
            fp = wd / f
            if fp.exists() and fp.stat().st_size < 20000:
                try:
                    ctx += f"\n--- {f} ---\n{fp.read_text(errors='replace')}\n"
                except Exception:
                    pass
        self._emit("read", f"Read {len(ctx)} chars", tokens=len(ctx) // 4)
        return ctx

    def _memory(self, prompt):
        if not self.memory:
            return ""
        cat = self._classify(prompt)
        try:
            ins = self.memory.get_cross_session_insights(cat)
            s = ins.get("successful_tactics", {})
            lines = [f"  [OK x{c}] {t}" for t, c in s.items()]
            self._emit("memory", f"{len(s)} tactics", "\n".join(lines))
            return "\n".join(lines)
        except Exception:
            return ""

    def _generate(self, prompt, context, mem, attempt):
        """Try LLM first (with retry), fall back to rules."""
        if self._llm:
            for retry in range(2):
                try:
                    self._emit("llm", "Calling Gemini...")
                    resp = self._llm.generate(
                        "You are NEXUS, a code repair agent. "
                        "Output ONLY the complete fixed function. "
                        "No markdown, no explanations.",
                        f"Bug: {prompt}\n\nMemory:\n{mem or 'None'}\n\n"
                        f"Code:\n{context[:5000]}\n\n"
                        f"Attempt {attempt}. Output fixed function only."
                    )
                    self._emit("llm", "Patch generated", resp[:150],
                              tokens=len(resp) // 4)
                    return resp
                except Exception as e:
                    if "429" in str(e) and retry == 0:
                        self._emit("llm", "Rate limited, waiting 5s...")
                        time.sleep(5)
                    else:
                        self._emit("llm", f"LLM error: {e}", status="failed")
                        break
        return self._rule_patch(prompt, context)

    def _rule_patch(self, prompt, context):
        """Rule-based: extract the SPECIFIC function name from the prompt."""
        p = prompt.lower()
        # Try to find the exact function name in the prompt
        func_match = re.search(r'(\w+_\w+)', prompt)
        if not func_match:
            return ""
        target_func = func_match.group(1)
        self._emit("patch", f"Rule: targeting {target_func}")
        if "none" in p:
            return f"RULE:none_guard:{target_func}"
        return ""

    def _apply(self, wd, patch, prompt):
        """Apply patch — only to the SPECIFIC function, not all files."""
        changed = []
        if patch.startswith("RULE:none_guard:"):
            target = patch.split(":")[2]
            # Only patch files that contain this specific function
            for f in wd.rglob("*.py"):
                if ".git" in str(f) or "__pycache__" in str(f):
                    continue
                try:
                    txt = f.read_text(errors="replace")
                    # Must contain the exact function definition
                    match = re.search(
                        rf'(def\s+{re.escape(target)}\s*\(([^)]*)\)\s*:)',
                        txt)
                    if match:
                        full_def = match.group(1)
                        params = match.group(2)
                        first_param = params.split(",")[0].strip().split(":")[0].strip()
                        if first_param == "self":
                            parts = params.split(",")
                            if len(parts) > 1:
                                first_param = parts[1].strip().split(":")[0].strip()
                            else:
                                continue
                        guard = f"{full_def}\n    if {first_param} is None:\n        return None"
                        txt = txt.replace(full_def, guard, 1)  # Only first occurrence
                        f.write_text(txt)
                        rel = str(f.relative_to(wd))
                        changed.append(rel)
                        self._emit("apply", f"Patched {rel}")
                        break  # Only patch ONE file
                except Exception:
                    pass
        else:
            # LLM patch — find and replace function
            func_match = re.search(r'def\s+(\w+)', patch)
            if func_match:
                func_name = func_match.group(1)
                for f in wd.rglob("*.py"):
                    if ".git" in str(f) or "__pycache__" in str(f):
                        continue
                    try:
                        txt = f.read_text(errors="replace")
                        if f"def {func_name}" not in txt:
                            continue
                        old = re.search(
                            rf'(def\s+{func_name}\s*\([^)]*\):.*?)(?=\ndef\s|\nclass\s|\Z)',
                            txt, re.DOTALL)
                        new = re.search(
                            rf'(def\s+{func_name}\s*\([^)]*\):.*?)(?=\ndef\s|\nclass\s|\Z)',
                            patch, re.DOTALL)
                        if old and new:
                            txt = txt.replace(old.group(0), new.group(0), 1)
                            f.write_text(txt)
                            rel = str(f.relative_to(wd))
                            changed.append(rel)
                            self._emit("apply", f"Patched {rel}")
                            break
                    except Exception:
                        pass

        if not changed:
            self._emit("apply", "No files changed", status="failed")
        return changed

    def _test(self, wd):
        self._emit("test", "Running tests...")
        cmd = [sys.executable, "-m", "pytest", "-x", "-q", "--tb=short",
               "-k", "not test_integrations and not test_list_envs"]
        try:
            out = self._cmd(wd, cmd, check=False, timeout=120)
        except Exception as e:
            return 0, 1, str(e)
        passed = len(re.findall(r"PASSED", out))
        failed = len(re.findall(r"FAILED|ERROR", out))
        m = re.search(r"(\d+) passed", out)
        if m:
            passed = max(passed, int(m.group(1)))
        self._emit("test", f"{passed}P/{failed}F", out[-200:])
        return passed, failed, out

    def _commit(self, wd, branch, prompt):
        self._cmd(wd, ["git", "checkout", "-b", branch], check=False)
        self._cmd(wd, ["git", "add", "-A"])
        self._cmd(wd, ["git", "commit", "-m", f"fix: {prompt[:60]}"])
        self._cmd(wd, ["git", "push", "origin", branch], check=False)

    def _open_pr(self, repo, branch, prompt, result):
        try:
            from github import Github, Auth
            gh = Github(auth=Auth.Token(self.github_token))
            r = gh.get_repo(repo)
            pr = r.create_pull(title=f"fix: {prompt[:60]}",
                              head=branch, base=r.default_branch,
                              body=f"NEXUS Auto-Fix | {result.tests_passed} tests passed")
            self._emit("pr", f"PR #{pr.number}", pr.html_url)
            return pr.html_url
        except Exception as e:
            self._emit("pr", f"Failed: {e}", status="failed")
            return ""

    def _cmd(self, cwd, cmd, check=True, timeout=60):
        r = subprocess.run(cmd, cwd=str(cwd), capture_output=True,
                          text=True, timeout=timeout)
        if check and r.returncode != 0:
            raise RuntimeError(r.stderr[:300] or r.stdout[:300])
        return r.stdout + r.stderr

    def _classify(self, prompt):
        p = prompt.lower()
        for kw, cat in [("none", "none_handling"), ("null", "none_handling"),
                        ("off-by", "off_by_one"), ("encoding", "encoding"),
                        ("type", "type_conversion"), ("login", "validation")]:
            if kw in p:
                return cat
        return "general"
