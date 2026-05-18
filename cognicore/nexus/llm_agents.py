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
    """Generates patches using a real LLM (Gemini/OpenAI/Claude)."""

    def __init__(self, llm: LLMClient = None):
        model_name = "rule-based"
        self.llm = llm or LLMClient(provider="auto")
        if self.llm.available:
            model_name = self.llm.model
        super().__init__(AgentRole.CODER, model_name)

    def _run(self, ctx: AgentContext) -> AgentResult:
        if not self.llm.available:
            return AgentResult(
                role=self.role, success=False,
                error="No LLM available", tokens_in=0, tokens_out=0)

        # Build prompt with CogniCore memory + reflection
        prompt = self._build_prompt(ctx)
        tokens_in = len(prompt) // 4  # rough estimate

        response = self.llm.generate(prompt, temperature=0.3)
        if not response:
            return AgentResult(
                role=self.role, success=False,
                error="LLM returned empty response",
                tokens_in=tokens_in, tokens_out=0)

        code = extract_code(response)
        tokens_out = len(response) // 4

        if not code or len(code.strip()) < 10:
            return AgentResult(
                role=self.role, success=False, output=response[:200],
                error="Could not extract valid code",
                artifacts={"tactic": "llm_fail", "raw_response": response[:300]},
                tokens_in=tokens_in, tokens_out=tokens_out)

        return AgentResult(
            role=self.role, success=True, output=code,
            artifacts={"tactic": "llm_repair", "model": self.llm.model},
            tokens_in=tokens_in, tokens_out=tokens_out)

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
