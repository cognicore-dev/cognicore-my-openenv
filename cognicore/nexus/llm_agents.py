"""
LLM-Backed Agents for NEXUS — real Gemini/OpenAI/Claude integration.
These replace rule-based agents with actual LLM reasoning.
"""
import re, time
from cognicore.nexus.agent import BaseAgent, AgentRole, AgentContext, AgentResult
from cognicore.research.llm_client import LLMClient


def extract_code(text: str) -> str:
    """Extract clean Python code from LLM output."""
    if not text:
        return ""
    # Try markdown code blocks first
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        code = blocks[0].strip()
    else:
        code = text.strip()
    # Strip line numbers like "1: ", "12: "
    lines = []
    for line in code.split('\n'):
        cleaned = re.sub(r'^\s*\d+[:\|]\s?', '', line)
        lines.append(cleaned)
    return '\n'.join(lines)


class LLMCoderAgent(BaseAgent):
    """Generates patches using a real LLM with rule-based fallback."""

    def __init__(self, llm: LLMClient = None, fix_db=None):
        model_name = "rule-based"
        self.llm = llm or LLMClient(provider="auto")
        self.fix_db = fix_db or {}
        if self.llm.available:
            model_name = self.llm.model
        super().__init__(AgentRole.CODER, model_name)
        self.llm_successes = 0
        self.llm_failures = 0

    def _run(self, ctx: AgentContext) -> AgentResult:
        # Try LLM first
        if self.llm.available:
            prompt = self._build_prompt(ctx)
            tokens_in = len(prompt) // 4

            response = self.llm.generate(prompt, temperature=0.3)
            if response:
                code = extract_code(response)
                tokens_out = len(response) // 4
                if code and len(code.strip()) >= 10:
                    self.llm_successes += 1
                    return AgentResult(
                        role=self.role, success=True, output=code,
                        artifacts={"tactic": "llm_repair", "model": self.llm.model},
                        tokens_in=tokens_in, tokens_out=tokens_out)

            self.llm_failures += 1

        # Fallback to rule-based fixes
        return self._rule_based_fallback(ctx)

    def _rule_based_fallback(self, ctx: AgentContext) -> AgentResult:
        """Fall back to rule-based fix database when LLM unavailable."""
        from cognicore.research.patch_intelligence import patch_hash
        fixes = self.fix_db.get(ctx.task_id, {})
        failed_hashes = {patch_hash(p.get("patch", "")) for p in ctx.previous_patches}

        disabled = set()
        for h in ctx.memory_hints:
            if "'guard' failed" in h:
                disabled.add("guard")

        strategy = "rewrite" if ctx.attempt > 1 else "guard"
        if strategy in disabled:
            strategy = "rewrite"

        order = [strategy] + [t for t in fixes if t != strategy and t not in disabled]
        for tactic in order:
            if tactic in fixes:
                try:
                    patch = fixes[tactic](ctx.buggy_code)
                    if patch and patch_hash(patch) not in failed_hashes:
                        return AgentResult(
                            role=self.role, success=True, output=patch,
                            artifacts={"tactic": f"rule:{tactic}"},
                            tokens_in=len(ctx.buggy_code), tokens_out=len(patch))
                except Exception:
                    continue

        return AgentResult(
            role=self.role, success=False, output=ctx.buggy_code,
            artifacts={"tactic": "exhausted"}, error="No viable fix",
            tokens_in=len(ctx.buggy_code), tokens_out=0)

    def _build_prompt(self, ctx: AgentContext) -> str:
        parts = [
            "You are an expert Python debugging agent.",
            "Fix the buggy code below so that the test passes.",
            "Return ONLY the corrected Python code. No explanations.",
            "Do NOT include test code in your response.",
            ""
        ]

        # Memory hints from CogniCore
        if ctx.memory_hints:
            parts.append("## Historical Context (from persistent memory)")
            for hint in ctx.memory_hints[:3]:
                parts.append(f"- {hint}")
            parts.append("")

        # Reflection
        if ctx.reflection:
            parts.append(f"## Reflection from past failures")
            parts.append(ctx.reflection[:200])
            parts.append("")

        # Previous failed attempts
        if ctx.previous_patches:
            parts.append(f"## Previous Failed Attempts ({len(ctx.previous_patches)})")
            for prev in ctx.previous_patches[-2:]:
                err = prev.get('error', 'unknown')[:100]
                parts.append(f"- Attempt {prev.get('attempt',0)}: {err}")
            parts.append("DO NOT repeat these approaches.")
            parts.append("")

        # Bug description
        parts.append(f"## Bug Description")
        parts.append(ctx.task_description[:300])
        parts.append("")

        # Buggy code
        parts.append("## Buggy Code")
        parts.append("```python")
        parts.append(ctx.buggy_code)
        parts.append("```")
        parts.append("")

        # Test code (for context)
        parts.append("## Test (must pass)")
        parts.append("```python")
        parts.append(ctx.test_code[:500])
        parts.append("```")
        parts.append("")

        parts.append("Return ONLY the fixed Python code, no markdown, no explanations.")
        return '\n'.join(parts)


class LLMReviewerAgent(BaseAgent):
    """Reviews patches using an LLM for semantic correctness checking."""

    def __init__(self, llm: LLMClient = None):
        model_name = "rule-based"
        self.llm = llm or LLMClient(provider="auto")
        if self.llm.available:
            model_name = self.llm.model
        super().__init__(AgentRole.REVIEWER, model_name)

    def _run(self, ctx: AgentContext) -> AgentResult:
        patch = ctx.metadata.get("current_patch", "")
        if not patch:
            return AgentResult(role=self.role, success=False,
                             output="No patch", tokens_in=0, tokens_out=0)

        if not self.llm.available:
            # Fallback: always approve (let tester catch issues)
            return AgentResult(role=self.role, success=True,
                             output="APPROVED (no LLM)", tokens_in=0, tokens_out=0)

        prompt = f"""Review this Python patch for correctness.
Bug: {ctx.task_description[:200]}

Original buggy code:
```python
{ctx.buggy_code}
```

Proposed fix:
```python
{patch}
```

Reply with ONLY one word: APPROVE or REJECT
If REJECT, add one sentence explaining why."""

        tokens_in = len(prompt) // 4
        response = self.llm.generate(prompt, temperature=0.1)
        tokens_out = len(response) // 4 if response else 0

        if response and "APPROVE" in response.upper():
            return AgentResult(role=self.role, success=True,
                             output="APPROVED by LLM",
                             tokens_in=tokens_in, tokens_out=tokens_out)

        return AgentResult(role=self.role, success=False,
                         output=response[:200] if response else "No response",
                         tokens_in=tokens_in, tokens_out=tokens_out)
