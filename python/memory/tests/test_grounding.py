"""Phase C — filesystem grounding tests.

Verifies the guard that demotes a candidate memory when it references
files or symbols that don't exist in the project.

Run: pytest aisys/memory/tests/test_grounding.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


# ----- extract_code_refs -----

@pytest.mark.parametrize("content,expected_count", [
    ("look at store.py for details", 1),
    ("the foo.bar() function does it", 1),
    ("see `aisys/memory/store.py`", 1),
    ("just plain text with no refs", 0),
    ("multiple files: store.py and schema.py exist", 2),
    ("path/to/file.ts is here", 1),
    ("Don't match foo or bar standalone", 0),
])
def test_extract_code_refs_count(content, expected_count):
    from memory.grounding import extract_code_refs
    refs = extract_code_refs(content)
    assert len(refs) == expected_count, f"got {refs}"


def test_extract_code_refs_strips_backticks():
    from memory.grounding import extract_code_refs
    refs = extract_code_refs("see `store.py` here")
    # Every returned ref should be filesystem-checkable (no backticks)
    for r in refs:
        assert "`" not in r


# ----- filesystem_exists -----

def test_filesystem_exists_for_real_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_DATA_DIR", str(tmp_path))
    real = tmp_path / "exists.py"
    real.write_text("# real")
    from memory.grounding import filesystem_exists
    assert filesystem_exists(str(real))


def test_filesystem_exists_returns_false_for_missing():
    from memory.grounding import filesystem_exists
    assert not filesystem_exists("definitely_does_not_exist_12345.py")


def test_filesystem_exists_resolves_relative_to_hermes_root(tmp_path, monkeypatch):
    """Relative paths get resolved against repo root."""
    monkeypatch.chdir(tmp_path)
    rel = Path("subdir/file.py")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file.py").write_text("# real")
    from memory.grounding import filesystem_exists
    assert filesystem_exists("subdir/file.py")
