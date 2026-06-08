"""
Concrete Agent Implementations for NEXUS.
Rule-based agents (Phase 1) + LLM-backed agents (Phase 2).
"""
import re, math
from cognicore.nexus.agent import BaseAgent, AgentRole, AgentContext, AgentResult
from cognicore.research.patch_intelligence import combined_similarity, patch_hash
from cognicore.research.sandbox import sandbox_fast, sandbox_isolated


# ══════════════════════════════════════════════════════════
# MEMORY AGENT — queries CogniCore persistent store
# ══════════════════════════════════════════════════════════
class MemoryAgent(BaseAgent):
    """Retrieves relevant past experiences from CogniCore memory."""

    def __init__(self, runtime=None, persistent_store=None):
        super().__init__(AgentRole.MEMORY, "rule-based")
        self.runtime = runtime
        self.persistent_store = persistent_store

    def _run(self, ctx: AgentContext) -> AgentResult:
        hints = []
        reflection = ""
        disabled_tactics = []

        # Cross-session persistent memory
        if self.persistent_store:
            category = ctx.metadata.get("category", "general")
            insights = self.persistent_store.get_cross_session_insights(category)
            for tactic, cnt in insights.get("failed_tactics", {}).items():
                if cnt >= 1:
                    disabled_tactics.append(tactic)
                    hints.append(f"Historical: '{tactic}' failed {cnt}x in '{category}'")

        # In-session memory
        if self.runtime:
            category = ctx.metadata.get("category", "general")
            mem_ctx = self.runtime._build_context(category)
            if mem_ctx.get("memory"):
                fails = [e for e in mem_ctx["memory"] if not e.get("correct")]
                if fails:
                    hints.append(f"{len(fails)} past failures in '{category}'")
            if mem_ctx.get("reflection_hint"):
                reflection = mem_ctx["reflection_hint"]

        return AgentResult(
            role=self.role, success=True,
            output=reflection,
            artifacts={
                "hints": hints,
                "reflection": reflection,
                "disabled_tactics": disabled_tactics,
            },
            tokens_in=50, tokens_out=len(str(hints)))


# ══════════════════════════════════════════════════════════
# PLANNER AGENT — decomposes task into strategy
# ══════════════════════════════════════════════════════════
class PlannerAgent(BaseAgent):
    """Analyzes the bug and selects repair strategy."""

    def __init__(self):
        super().__init__(AgentRole.PLANNER, "rule-based")

    def _run(self, ctx: AgentContext) -> AgentResult:
        # Analyze error type from previous attempts
        error_types = set()
        for prev in ctx.previous_patches:
            err = prev.get("error", "")
            for etype in ["TypeError", "ValueError", "KeyError", "IndexError",
                          "AttributeError", "AssertionError", "RecursionError",
                          "TimeoutError", "NoneType"]:
                if etype in err:
                    error_types.add(etype)

        # Select strategy based on error analysis + memory
        disabled = set(ctx.metadata.get("disabled_tactics", []))
        if ctx.memory_hints:
            for h in ctx.memory_hints:
                if "'guard' failed" in h:
                    disabled.add("guard")

        if ctx.attempt == 1 and "guard" not in disabled:
            strategy = "guard"
            plan = "Apply minimal guard fix (null check, type check, boundary fix)"
        elif ctx.attempt <= 2 or "guard" in disabled:
            strategy = "rewrite"
            plan = "Rewrite the function with correct logic"
        else:
            strategy = "restructure"
            plan = "Restructure the code architecture"

        if error_types:
            plan += f" | Errors seen: {', '.join(error_types)}"
        if ctx.reflection:
            plan += f" | Reflection: {ctx.reflection[:100]}"

        return AgentResult(
            role=self.role, success=True, output=plan,
            artifacts={"plan": plan, "strategy": strategy, "disabled": list(disabled)},
            tokens_in=len(ctx.task_description), tokens_out=len(plan))


# ══════════════════════════════════════════════════════════
# CODER AGENT — generates patches
# ══════════════════════════════════════════════════════════
class CoderAgent(BaseAgent):
    """Generates code patches. Rule-based (Phase 1) or LLM-backed (Phase 2)."""

    def __init__(self, fix_db=None):
        super().__init__(AgentRole.CODER, "rule-based")
        self.fix_db = fix_db or {}

    def _run(self, ctx: AgentContext) -> AgentResult:
        strategy = ctx.metadata.get("plan_strategy",
                    ctx.metadata.get("plan", {}).get("strategy") if isinstance(ctx.metadata.get("plan"), dict) else None)
        # Extract strategy from plan artifacts
        plan_data = ctx.metadata.get("plan", "")
        if isinstance(plan_data, str):
            if "rewrite" in plan_data.lower() or "restructure" in plan_data.lower():
                strategy = "rewrite"
            elif "guard" in plan_data.lower():
                strategy = "guard"

        disabled = set()
        for h in ctx.memory_hints:
            if "'guard' failed" in h:
                disabled.add("guard")

        if strategy in disabled:
            strategy = "rewrite"
        if not strategy:
            strategy = "rewrite" if ctx.attempt > 1 else "guard"

        fixes = self.fix_db.get(ctx.task_id, {})
        failed_hashes = {patch_hash(p.get("patch", "")) for p in ctx.previous_patches}

        # Try strategies in preference order
        order = [strategy] + [t for t in fixes if t != strategy and t not in disabled]
        for tactic in order:
            if tactic in fixes:
                try:
                    patch = fixes[tactic](ctx.buggy_code)
                    if patch and patch_hash(patch) not in failed_hashes:
                        return AgentResult(
                            role=self.role, success=True, output=patch,
                            artifacts={"tactic": tactic},
                            tokens_in=len(ctx.buggy_code), tokens_out=len(patch))
                except Exception:
                    continue

        return AgentResult(
            role=self.role, success=False, output=ctx.buggy_code,
            artifacts={"tactic": "exhausted"}, error="No viable fix found",
            tokens_in=len(ctx.buggy_code), tokens_out=0)


# ══════════════════════════════════════════════════════════
# REVIEWER AGENT — validates patch quality
# ══════════════════════════════════════════════════════════
class ReviewerAgent(BaseAgent):
    """Reviews patches for quality. Checks similarity to past failures."""

    def __init__(self, similarity_threshold: float = 0.85):
        super().__init__(AgentRole.REVIEWER, "rule-based")
        self.threshold = similarity_threshold

    def _run(self, ctx: AgentContext) -> AgentResult:
        patch = ctx.metadata.get("current_patch", "")
        if not patch:
            return AgentResult(role=self.role, success=False,
                             output="No patch to review", tokens_in=0, tokens_out=0)

        # Check similarity to previously failed PATCHES (not buggy code)
        # Only reject if the patch is near-identical to a previously tried fix
        rejected = False
        for prev in ctx.previous_patches[-3:]:
            prev_patch = prev.get("patch", "")
            if prev_patch and len(prev_patch) > 20:
                sim = combined_similarity(patch, prev_patch)
                # Only reject if VERY similar (>95%) to a failed fix attempt
                # AND the patch is different from the buggy code
                buggy_sim = combined_similarity(patch, ctx.buggy_code)
                if sim > 0.95 and buggy_sim < 0.90:
                    return AgentResult(
                        role=self.role, success=False,
                        output=f"REJECTED: {sim:.0%} identical to failed attempt {prev.get('attempt',0)}",
                        artifacts={"similarity": sim, "rejected": True},
                        tokens_in=len(patch), tokens_out=50)

        # Basic quality checks
        if len(patch.strip()) < 10:
            return AgentResult(role=self.role, success=False,
                             output="REJECTED: Patch too short", tokens_in=len(patch), tokens_out=20)

        # Check patch is actually different from buggy code
        buggy_sim = combined_similarity(patch, ctx.buggy_code)
        if buggy_sim > 0.99:
            return AgentResult(role=self.role, success=False,
                             output="REJECTED: Patch identical to buggy code",
                             artifacts={"no_change": True},
                             tokens_in=len(patch), tokens_out=20)

        return AgentResult(
            role=self.role, success=True,
            output="APPROVED: Patch passes review",
            artifacts={"approved": True},
            tokens_in=len(patch), tokens_out=20)


# ══════════════════════════════════════════════════════════
# TESTER AGENT — executes tests in sandbox
# ══════════════════════════════════════════════════════════
class TesterAgent(BaseAgent):
    """Executes patches against test suite in isolated sandbox."""

    def __init__(self, isolated: bool = False):
        super().__init__(AgentRole.TESTER, "sandbox")
        self.isolated = isolated

    def _run(self, ctx: AgentContext) -> AgentResult:
        patch = ctx.metadata.get("current_patch", "")
        if not patch:
            return AgentResult(role=self.role, success=False,
                             error="No patch to test", tokens_in=0, tokens_out=0)

        if self.isolated:
            ok, err, meta = sandbox_isolated(patch, ctx.test_code, timeout=30)
            return AgentResult(
                role=self.role, success=ok, output="PASS" if ok else "FAIL",
                error=err or "", artifacts={"exec_meta": meta},
                tokens_in=len(patch) + len(ctx.test_code), tokens_out=50)
        else:
            ok, err = sandbox_fast(patch, ctx.test_code)
            return AgentResult(
                role=self.role, success=ok, output="PASS" if ok else "FAIL",
                error=err or "",
                tokens_in=len(patch) + len(ctx.test_code), tokens_out=50)


# ══════════════════════════════════════════════════════════
# VERIFIER AGENT — independent safety verification
# ══════════════════════════════════════════════════════════
class VerifierAgent(BaseAgent):
    """Independent verification of patches before deployment."""

    def __init__(self):
        super().__init__(AgentRole.VERIFIER, "rule-based")

    def _run(self, ctx: AgentContext) -> AgentResult:
        patch = ctx.metadata.get("current_patch", "")
        issues = []

        # Diff size check
        if len(patch) > 5000:
            issues.append("Patch exceeds 5000 chars — manual review required")

        # Dangerous pattern detection
        dangerous = ["os.system", "subprocess.call", "eval(", "__import__",
                      "exec(", "open(", "shutil.rmtree"]
        for d in dangerous:
            if d in patch:
                issues.append(f"Dangerous pattern: {d}")

        if issues:
            return AgentResult(
                role=self.role, success=False,
                output="; ".join(issues), artifacts={"issues": issues},
                tokens_in=len(patch), tokens_out=100)

        return AgentResult(
            role=self.role, success=True,
            output="VERIFIED: Safe to deploy",
            tokens_in=len(patch), tokens_out=20)
