"""
Atomic memory decomposition — zero LLM calls.

Splits paragraphs and compound sentences into atomic facts
so each fact is independently searchable. This is the key
advantage Mem0 has over raw-text storage, implemented here
without any API calls.
"""

import re
from typing import List


# Sentence boundary regex: period/exclamation/question followed by space+capital
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# Common conjunctions / list separators that indicate compound facts
_COMPOUND = re.compile(
    r'\s*(?:,\s*(?:and|but|also|plus|as well as)|;\s*|,\s+)\s*',
    re.IGNORECASE
)

# Minimum length (chars) for a fact to be worth storing independently
_MIN_FACT_LEN = 12

# Maximum length (chars) before we try to decompose
_MAX_ATOMIC_LEN = 200


def decompose(text: str) -> List[str]:
    """Break text into atomic facts without any LLM call.

    Strategy:
    1. If the text is already short and atomic, return as-is.
    2. Split on sentence boundaries first.
    3. For each sentence, split on compound conjunctions.
    4. Deduplicate and filter out fragments that are too short.

    Examples:
        >>> decompose("CogniCore v0.9.3, 7000 downloads, built by Kaushalt in Mumbai")
        ['CogniCore v0.9.3', '7000 downloads', 'built by Kaushalt in Mumbai']

        >>> decompose("User prefers dark mode")
        ['User prefers dark mode']
    """
    text = text.strip()
    if not text:
        return []

    # Already atomic — short, single fact
    if len(text) <= _MAX_ATOMIC_LEN and not _SENT_SPLIT.search(text) and ',' not in text and ';' not in text:
        return [text]

    # Phase 1: sentence splitting
    sentences = _SENT_SPLIT.split(text)

    # Phase 2: compound splitting within each sentence
    facts = []
    for sent in sentences:
        sent = sent.strip().rstrip('.')
        if not sent:
            continue

        # Try compound split
        parts = _COMPOUND.split(sent)
        parts = [p.strip().rstrip('.').strip() for p in parts if p and p.strip()]

        if len(parts) > 1:
            # Validate each part is meaningful
            for part in parts:
                if len(part) >= _MIN_FACT_LEN:
                    facts.append(part)
                elif facts:
                    # Short fragment — attach to previous fact
                    facts[-1] = f"{facts[-1]}, {part}"
        else:
            facts.append(sent)

    # Phase 3: deduplicate (preserve order)
    seen = set()
    unique = []
    for f in facts:
        key = f.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique if unique else [text]
