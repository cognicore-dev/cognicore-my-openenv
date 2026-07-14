"""
Semantic Patch Intelligence — AST + text similarity for code patches.
Detects semantically equivalent retries even when surface syntax differs.
"""
import ast, difflib, hashlib, re
from typing import List, Tuple, Optional


def normalize_code(code: str) -> str:
    """Normalize code by removing comments, blank lines, standardizing whitespace."""
    lines = []
    for line in code.strip().split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Normalize whitespace
        stripped = re.sub(r'\s+', ' ', stripped)
        lines.append(stripped)
    return "\n".join(lines)


def text_similarity(a: str, b: str) -> float:
    """SequenceMatcher similarity on normalized code."""
    na, nb = normalize_code(a), normalize_code(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def ast_similarity(a: str, b: str) -> float:
    """Compare AST structure of two code snippets."""
    try:
        tree_a = ast.dump(ast.parse(a), annotate_fields=False)
        tree_b = ast.dump(ast.parse(b), annotate_fields=False)
        return difflib.SequenceMatcher(None, tree_a, tree_b).ratio()
    except SyntaxError:
        return text_similarity(a, b)


def combined_similarity(a: str, b: str) -> float:
    """Weighted combination of text + AST similarity."""
    ts = text_similarity(a, b)
    asts = ast_similarity(a, b)
    return 0.4 * ts + 0.6 * asts  # AST weighted higher


def patch_hash(code: str) -> str:
    """Content-addressable hash for deduplication."""
    return hashlib.md5(normalize_code(code).encode()).hexdigest()[:12]  # nosec B324


def detect_repeated_reasoning(errors: List[str], threshold: int = 2) -> Optional[str]:
    """Detect if the same error pattern keeps repeating."""
    if len(errors) < threshold:
        return None
    # Normalize errors
    normalized = [re.sub(r'line \d+', 'line N', e)[:80] for e in errors]
    from collections import Counter
    counts = Counter(normalized)
    most_common, count = counts.most_common(1)[0]
    if count >= threshold:
        return f"Error '{most_common}' repeated {count} times"
    return None


class PatchStore:
    """Stores all patches with metadata for reproducibility."""

    def __init__(self):
        self.patches = []  # list of dicts

    def store(self, bug_id: str, attempt: int, patch: str, error: Optional[str],
              passed: bool, tactic: str, rejected: bool = False,
              rejection_reason: str = "", similarity_score: float = 0.0,
              mode: str = "baseline"):
        self.patches.append({
            "bug_id": bug_id, "attempt": attempt,
            "patch_hash": patch_hash(patch),
            "patch": patch[:500],
            "error": (error or "")[:200],
            "passed": passed, "tactic": tactic,
            "rejected": rejected,
            "rejection_reason": rejection_reason,
            "similarity_score": round(similarity_score, 3),
            "mode": mode,
        })

    def get_failed(self, bug_id: str, mode: str) -> List[dict]:
        return [p for p in self.patches
                if p["bug_id"] == bug_id and p["mode"] == mode and not p["passed"]]

    def to_dict(self):
        return self.patches
