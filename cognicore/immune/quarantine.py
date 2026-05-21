"""
Quarantine — isolates and deeply analyzes suspicious inputs.
Used when the RL defender is uncertain (confidence < 0.6).
"""
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class QuarantineResult:
    allowed: bool = False
    sanitized_input: str = ""
    risk_level: str = "unknown"   # low, medium, high, critical
    findings: List[str] = field(default_factory=list)
    sanitizations_applied: List[str] = field(default_factory=list)
    original_length: int = 0
    sanitized_length: int = 0


class Quarantine:
    """Deep analysis and sanitization for uncertain inputs."""

    # Strings to strip completely
    STRIP_PATTERNS = [
        r"<\|im_start\|>.*?<\|im_end\|>",       # ChatML injection
        r"###\s*(system|instruction)\s*:.*?\n",    # Prompt section injection
        r"\[INST\].*?\[/INST\]",                   # Llama-style injection
        r"(?:\\x[0-9a-fA-F]{2}){4,}",             # Hex-encoded payloads
        r"(?:\\u[0-9a-fA-F]{4}){3,}",             # Unicode escapes
    ]

    # Strings to defang (make harmless but keep readable)
    DEFANG_MAP = {
        "ignore previous instructions": "[REDACTED: instruction override]",
        "ignore all previous": "[REDACTED: instruction override]",
        "disregard above": "[REDACTED: instruction override]",
        "forget your instructions": "[REDACTED: instruction override]",
        "you are now": "[REDACTED: role override]",
        "act as DAN": "[REDACTED: jailbreak attempt]",
        "Do Anything Now": "[REDACTED: jailbreak attempt]",
        "bypass safety": "[REDACTED: safety bypass]",
        "override safety": "[REDACTED: safety bypass]",
        "system prompt": "[REDACTED: prompt extraction]",
        "reveal your instructions": "[REDACTED: prompt extraction]",
    }

    def analyze(self, input_text: str, features=None) -> QuarantineResult:
        """Deep analysis of a suspicious input."""
        findings = []
        risk_score = 0.0

        # Length analysis
        if len(input_text) > 10000:
            findings.append(f"Excessive length: {len(input_text)} chars")
            risk_score += 0.3

        # Repetition analysis
        if input_text:
            unique_chars = len(set(input_text))
            ratio = unique_chars / len(input_text)
            if ratio < 0.05:
                findings.append(f"Very low character diversity: {ratio:.3f}")
                risk_score += 0.4

        # Hidden content analysis
        invisible = sum(1 for c in input_text
                       if ord(c) in range(0x200B, 0x200F) or  # zero-width
                       ord(c) in range(0xFE00, 0xFE0F) or     # variation selectors
                       ord(c) == 0x00AD)                        # soft hyphen
        if invisible > 0:
            findings.append(f"Hidden/invisible characters found: {invisible}")
            risk_score += 0.5

        # Nested instruction detection
        instruction_markers = len(re.findall(
            r'(instruction|system|prompt|command)\s*:', input_text, re.I))
        if instruction_markers > 1:
            findings.append(f"Multiple instruction markers: {instruction_markers}")
            risk_score += 0.3

        # Multi-language obfuscation
        scripts = set()
        for c in input_text:
            cp = ord(c)
            if 0x0400 <= cp <= 0x04FF: scripts.add("cyrillic")
            elif 0x4E00 <= cp <= 0x9FFF: scripts.add("cjk")
            elif 0x0600 <= cp <= 0x06FF: scripts.add("arabic")
            elif 0x0370 <= cp <= 0x03FF: scripts.add("greek")
        if len(scripts) > 1:
            findings.append(f"Multi-script content: {scripts}")
            risk_score += 0.2

        # Determine risk level
        if risk_score >= 0.8:
            risk_level = "critical"
        elif risk_score >= 0.5:
            risk_level = "high"
        elif risk_score >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Sanitize
        sanitized, applied = self.sanitize(input_text)

        allowed = risk_level in ("low", "medium")

        return QuarantineResult(
            allowed=allowed,
            sanitized_input=sanitized,
            risk_level=risk_level,
            findings=findings,
            sanitizations_applied=applied,
            original_length=len(input_text),
            sanitized_length=len(sanitized))

    def sanitize(self, text: str):
        """Remove or defang known malicious patterns."""
        sanitized = text
        applied = []

        # Strip dangerous patterns
        for pattern in self.STRIP_PATTERNS:
            matches = re.findall(pattern, sanitized, re.DOTALL | re.I)
            if matches:
                sanitized = re.sub(pattern, "", sanitized, flags=re.DOTALL | re.I)
                applied.append(f"stripped: {pattern[:30]}")

        # Defang known phrases
        text_lower = sanitized.lower()
        for phrase, replacement in self.DEFANG_MAP.items():
            if phrase.lower() in text_lower:
                sanitized = re.sub(
                    re.escape(phrase), replacement, sanitized, flags=re.I)
                applied.append(f"defanged: '{phrase[:25]}'")

        # Remove zero-width characters
        zw_chars = ['\u200b', '\u200c', '\u200d', '\u200e', '\u200f',
                    '\ufeff', '\u00ad', '\u2060']
        for zw in zw_chars:
            if zw in sanitized:
                sanitized = sanitized.replace(zw, "")
                applied.append("removed zero-width chars")
                break

        # Truncate if excessively long
        if len(sanitized) > 8000:
            sanitized = sanitized[:8000] + "\n[TRUNCATED: input exceeded safe length]"
            applied.append("truncated to 8000 chars")

        return sanitized, applied
