"""Secret scrubbing — strip credentials from memory content before storage.

Memory pasted from terminal output, error messages, or shell scripts may
contain API keys, AWS credentials, GitHub tokens. We MUST scrub these
before they land in L2 where they could resurface via retrieval months
later (and possibly leak to other AI tools).

Patterns are conservative — false positives are better than false
negatives for this purpose. Each match becomes [REDACTED:<kind>:<len>]
so the structure of the content is preserved.
"""

from __future__ import annotations

import re
from typing import Pattern


# Each entry: (kind, regex). Order matters for overlapping patterns.
# Patterns are conservative — they aim for high-precision detection of
# *actual* credentials, not just any token-ish string.
SECRET_PATTERNS: list[tuple[str, Pattern[str]]] = [
    # Anthropic API keys
    ("anthropic", re.compile(r"sk-ant-api\d+-[\w\-]{40,}")),
    # OpenAI API keys (modern + legacy)
    ("openai", re.compile(r"sk-(?:proj-)?[A-Za-z0-9]{40,}")),
    # GitHub tokens
    ("github", re.compile(r"ghp_[A-Za-z0-9]{36,}")),
    ("github", re.compile(r"github_pat_[A-Za-z0-9_]{40,}")),
    # Slack tokens
    ("slack", re.compile(r"xox[bpasoe]-[A-Za-z0-9\-]{20,}")),
    # AWS access key ids (always start with AKIA, ASIA, etc., 20 chars)
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA|AIDA|AGPA|AROA|AIPA|ANPA|ANVA)[A-Z0-9]{16}\b")),
    # AWS secret keys (40 chars, mixed case + slashes/+ in base64)
    ("aws_secret_key", re.compile(r"aws_secret_access_key\s*=\s*([A-Za-z0-9/+=]{40})")),
    # Generic bearer tokens (JWT-like or long opaque strings)
    ("bearer", re.compile(r"Bearer\s+([A-Za-z0-9_\-\.]{30,})")),
    # Google API keys
    ("google", re.compile(r"AIza[A-Za-z0-9\-_]{35}")),
    # Stripe keys
    ("stripe", re.compile(r"(?:sk|pk|rk)_(?:test|live)_[A-Za-z0-9]{24,}")),
]


REDACTED_TEMPLATE = "[REDACTED:{kind}:{length}]"


def contains_secrets(text: str) -> bool:
    """Return True if any known credential pattern matches."""
    if not text:
        return False
    for _, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return True
    return False


def scrub(text: str) -> str:
    """Replace every detected secret with a length-tagged redaction marker.

    Idempotent — running twice produces same result (markers don't match).
    """
    if not text:
        return text
    result = text
    for kind, pattern in SECRET_PATTERNS:
        def _replace(match):
            full = match.group(0)
            return REDACTED_TEMPLATE.format(kind=kind, length=len(full))
        result = pattern.sub(_replace, result)
    return result


def scrub_with_count(text: str) -> tuple[str, int]:
    """Like scrub() but also returns the number of redactions made."""
    if not text:
        return text, 0
    count = 0
    result = text
    for kind, pattern in SECRET_PATTERNS:
        def _replace(match):
            nonlocal count
            count += 1
            full = match.group(0)
            return REDACTED_TEMPLATE.format(kind=kind, length=len(full))
        result = pattern.sub(_replace, result)
    return result, count
