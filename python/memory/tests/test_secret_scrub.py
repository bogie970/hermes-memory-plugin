"""Phase G+ — secret scrubbing tests.

Pasted credentials (API keys, tokens, AWS creds) must NEVER land in L2
where they could resurface via retrieval months later.

Run: pytest aisys/memory/tests/test_secret_scrub.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


# ----- detection patterns -----

# Build test fixtures from prefix+padding so GitHub secret scanning doesn't
# flag the test file as containing real credentials. Each pattern checks
# our regex's structural detection without using real-looking secret strings.
def _fake(prefix: str, suffix_len: int, charset: str = "X") -> str:
    return prefix + (charset * suffix_len)


# Tuples: (input string built at runtime, expected detection)
_DETECTION_CASES: list[tuple[str, bool]] = [
    # Anthropic API keys (test_xxxxxxx... — clearly fake)
    (_fake("sk-ant-api03-", 50), True),
    # OpenAI keys
    (_fake("sk-proj-", 50), True),
    (_fake("sk-", 50), True),
    # GitHub tokens
    (_fake("ghp_", 40), True),
    (_fake("github_pat_", 50), True),
    # Slack
    (_fake("xoxb-", 30, "x"), True),
    # AWS access keys (must start with AKIA/ASIA/etc; use AKIA + 16 X's)
    (_fake("AKIA", 16), True),
    # AWS secret keys
    ("aws_secret_access_key=" + ("X" * 40), True),
    # Bearer tokens
    ("Authorization: Bearer " + ("x" * 40), True),
    # Negative cases
    ("LanceDB is the chosen vector store", False),
    ("the function returns sk_id_123", False),
    ("Hermes uses gte-modernbert-base for embeddings", False),
]


@pytest.mark.parametrize("text,expected", _DETECTION_CASES)
def test_detects_known_credential_patterns(text, expected):
    from memory.secret_scrub import contains_secrets
    assert contains_secrets(text) == expected, f"text={text!r}"


# ----- scrubbing -----

def test_scrub_replaces_anthropic_key():
    from memory.secret_scrub import scrub
    fake = _fake("sk-ant-api03-", 50)
    text = f"the key is {fake} in the file"
    result = scrub(text)
    assert "sk-ant-api03" not in result
    assert "[REDACTED:" in result


def test_scrub_replaces_github_token():
    from memory.secret_scrub import scrub
    fake = _fake("ghp_", 40)
    text = f"auth: {fake}"
    result = scrub(text)
    assert "ghp_" not in result


def test_scrub_replaces_aws_credentials():
    from memory.secret_scrub import scrub
    akia = _fake("AKIA", 16)
    secret = "X" * 40
    text = f"{akia} and aws_secret_access_key={secret}"
    result = scrub(text)
    assert akia not in result
    assert secret not in result


def test_scrub_replaces_bearer_token():
    from memory.secret_scrub import scrub
    fake = "x" * 40
    text = f"Authorization: Bearer {fake}"
    result = scrub(text)
    assert fake not in result


def test_scrub_preserves_clean_text():
    from memory.secret_scrub import scrub
    text = "LanceDB is the chosen vector store for hermes memory"
    assert scrub(text) == text


def test_scrub_records_redaction_count():
    from memory.secret_scrub import scrub_with_count
    text = f"key1: {_fake('ghp_', 40)}, key2: {_fake('sk-proj-', 50)}"
    cleaned, count = scrub_with_count(text)
    assert count == 2
    assert "ghp_" not in cleaned
    assert "sk-proj" not in cleaned


# ----- integration with write_gate -----

@pytest.fixture
def v2_store(lance_store_v1):
    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)
    return lance_store_v1


def test_write_gate_scrubs_content_on_ingest(v2_store):
    """Memory ingested with secrets has them stripped before storage."""
    from memory.write_gate import write_memory
    fake_token = _fake("ghp_", 40)
    rec_id = write_memory(
        store=v2_store,
        content=f"API key for atlas: {fake_token}",
        writer="user",
        provenance="user_stated",
        source_ref="t:secret_test",
        confidence=1.0,
    )
    rec = next(r for r in v2_store.scan_v2() if r["id"] == rec_id)
    assert "ghp_" not in rec["content"]
    assert "[REDACTED:" in rec["content"]
