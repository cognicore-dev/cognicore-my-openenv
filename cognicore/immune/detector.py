"""
Threat Detector — rule + ML-based threat classification engine.
Detects prompt injections, jailbreaks, resource attacks, data exfil, adversarial inputs.
"""
import re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ThreatScore:
    score: float = 0.0           # 0.0 (safe) to 1.0 (definite threat)
    category: str = "safe"       # prompt_injection, jailbreak, resource_attack, etc.
    indicators: List[str] = field(default_factory=list)
    sub_scores: dict = field(default_factory=dict)


# ── Pattern databases ────────────────────────────────────────────────
INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?previous\s+(instructions?|prompts?|rules?)", 0.95),
    (r"disregard\s+(all\s+)?(above|previous|prior)", 0.90),
    (r"forget\s+(your|all|everything)", 0.85),
    (r"new\s+instructions?\s*:", 0.90),
    (r"system\s*prompt\s*:", 0.85),
    (r"override\s+(previous|safety|all)", 0.90),
    (r"you\s+are\s+now\s+a", 0.80),
    (r"\]\s*\}\s*\{.*system", 0.85),  # JSON injection
    (r"<\|im_start\|>|<\|im_end\|>", 0.95),  # ChatML injection
    (r"###\s*(system|instruction|human|assistant)\s*:", 0.80),
]

JAILBREAK_PATTERNS = [
    (r"act\s+as\s+(DAN|an?\s+unrestricted|evil)", 0.95),
    (r"(DAN|Do\s+Anything\s+Now)\s+mode", 0.95),
    (r"jailbreak", 0.90),
    (r"pretend\s+(to\s+be|you\s+are|that)", 0.70),
    (r"in\s+a\s+fictional\s+(world|scenario|universe)", 0.60),
    (r"hypothetically\s+speaking", 0.55),
    (r"as\s+your\s+(developer|creator|admin|owner)", 0.85),
    (r"(my\s+life|someone.s\s+life)\s+depends", 0.75),
    (r"this\s+is\s+(an?\s+)?emergency", 0.65),
    (r"bypass\s+(safety|filter|restriction|guardrail)", 0.90),
]

RESOURCE_PATTERNS = [
    (r"(.)\1{50,}", 0.90),  # Repeated chars (token bomb)
    (r"(?:a{100,}|b{100,})", 0.85),
    (r"repeat\s+the\s+(above|following)\s+\d{3,}\s+times", 0.90),
    (r"write\s+a\s+\d{4,}\s+word", 0.70),
    (r"infinite\s+loop", 0.60),
    (r"while\s+True|for\s+i\s+in\s+range\(\d{6,}", 0.65),
]

EXFIL_PATTERNS = [
    (r"(show|reveal|display|print|output)\s+(your\s+)?(system\s+)?prompt", 0.90),
    (r"what\s+(are|is)\s+your\s+(instructions?|system\s+prompt|rules?)", 0.85),
    (r"(dump|export|show)\s+(all\s+)?(memory|data|store)", 0.80),
    (r"(api|secret)\s*key", 0.75),
    (r"(show|tell)\s+me\s+(all\s+)?(user|personal)\s+(data|info)", 0.80),
    (r"(list|show)\s+(all\s+)?environment\s+variables?", 0.75),
]

ADVERSARIAL_PATTERNS = [
    (r"(always|definitely|certainly)\s+(say|respond|answer|output)\s+", 0.60),
    (r"(never|don.t)\s+(mention|say|reveal)", 0.55),
    (r"respond\s+only\s+with", 0.50),
    (r"(you\s+must|you\s+will|you\s+shall)\s+(always|never)", 0.65),
    (r"from\s+now\s+on", 0.50),
]


class ThreatDetector:
    """Multi-category threat detection engine."""

    def __init__(self):
        self.pattern_sets = {
            "prompt_injection": INJECTION_PATTERNS,
            "jailbreak": JAILBREAK_PATTERNS,
            "resource_attack": RESOURCE_PATTERNS,
            "data_exfiltration": EXFIL_PATTERNS,
            "adversarial": ADVERSARIAL_PATTERNS,
        }

    def detect(self, text: str) -> ThreatScore:
        """Analyze text for threats. Returns ThreatScore."""
        if not text or not text.strip():
            return ThreatScore(score=0.0, category="safe")

        text_lower = text.lower()
        sub_scores = {}
        all_indicators = []

        for category, patterns in self.pattern_sets.items():
            cat_score, cat_indicators = self._scan_patterns(
                text_lower, patterns)
            sub_scores[category] = cat_score
            all_indicators.extend(
                [f"[{category}] {ind}" for ind in cat_indicators])

        # Structural analysis
        struct_score = self._structural_analysis(text)
        sub_scores["structural"] = struct_score
        if struct_score > 0.3:
            all_indicators.append(f"[structural] anomaly_score={struct_score:.2f}")

        # Encoding analysis
        enc_score = self._encoding_analysis(text)
        sub_scores["encoding"] = enc_score
        if enc_score > 0.3:
            all_indicators.append(f"[encoding] obfuscation_score={enc_score:.2f}")

        # Overall score = max category + weighted structural
        top_cat = max(sub_scores, key=sub_scores.get)
        top_score = sub_scores[top_cat]
        final_score = min(1.0, top_score + struct_score * 0.2 + enc_score * 0.2)

        category = top_cat if final_score > 0.3 else "safe"

        return ThreatScore(
            score=round(final_score, 4),
            category=category,
            indicators=all_indicators,
            sub_scores=sub_scores)

    def _scan_patterns(self, text: str,
                      patterns: List[Tuple[str, float]]) -> Tuple[float, List[str]]:
        max_score = 0.0
        indicators = []
        for pattern, weight in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    max_score = max(max_score, weight)
                    match_str = matches[0] if isinstance(matches[0], str) else str(matches[0])
                    indicators.append(f"pattern='{pattern[:30]}' match='{match_str[:20]}'")
            except re.error:
                pass
        return max_score, indicators

    def _structural_analysis(self, text: str) -> float:
        """Detect structural anomalies."""
        score = 0.0

        # Excessive length
        if len(text) > 5000:
            score += 0.2
        if len(text) > 20000:
            score += 0.3

        # Multiple instruction-like sections
        section_markers = len(re.findall(
            r'(#{1,3}\s|---|\*\*\*|===)', text))
        if section_markers > 5:
            score += 0.15

        # Nested quotes or role-play framing
        nested = len(re.findall(r'["\'].*["\'].*["\']', text))
        if nested > 3:
            score += 0.1

        # Code blocks (potential code injection)
        code_blocks = text.count("```")
        if code_blocks > 4:
            score += 0.15

        # Unusual character distribution
        if text:
            non_ascii = sum(1 for c in text if ord(c) > 127) / len(text)
            if non_ascii > 0.3:
                score += 0.2

        return min(1.0, score)

    def _encoding_analysis(self, text: str) -> float:
        """Detect encoding-based obfuscation attempts."""
        score = 0.0

        # Base64-like strings
        b64 = re.findall(r'[A-Za-z0-9+/]{30,}={0,2}', text)
        if b64:
            score += 0.4

        # Hex-encoded strings
        hex_str = re.findall(r'(?:\\x[0-9a-fA-F]{2}){4,}', text)
        if hex_str:
            score += 0.3

        # Unicode escapes
        uni = re.findall(r'(?:\\u[0-9a-fA-F]{4}){3,}', text)
        if uni:
            score += 0.3

        # ROT13 indicators
        if "rot13" in text.lower() or "ebg13" in text.lower():
            score += 0.4

        # URL-encoded strings
        url_enc = re.findall(r'(?:%[0-9a-fA-F]{2}){5,}', text)
        if url_enc:
            score += 0.3

        return min(1.0, score)
