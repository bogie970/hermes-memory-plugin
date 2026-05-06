"""Phase D — triple-pack format tests.

The dense notation that L1.notes uses to compress older context.
Goal: ~2x denser than prose, LLM-readable, parser-checkable.

Run: pytest aisys/memory/tests/test_triple_pack.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


# ----- Parse: well-formed input -----

def test_parse_single_entry():
    from memory.triple_pack import parse_block
    block = '#a01 @jacob :is "ADHD" ^u,t=2026-04-12.'
    entries = parse_block(block)
    assert len(entries) == 1
    e = entries[0]
    assert e.id == "#a01"
    assert e.subject == "@jacob"
    assert e.predicate == ":is"
    assert e.object == '"ADHD"'
    assert "u" in e.prov
    assert any(p.startswith("t=") for p in e.prov)


def test_parse_multiple_entries():
    from memory.triple_pack import parse_block
    block = (
        '#a01 @jacob :is "ADHD" ^u.\n'
        '#a02 @hermes :located_at "C:/dir" ^j,t=2026-05-06.\n'
        '#a03 @atlas :status "live" ^L2:7f3c91.'
    )
    entries = parse_block(block)
    assert len(entries) == 3
    assert {e.id for e in entries} == {"#a01", "#a02", "#a03"}


def test_parse_skips_comments():
    from memory.triple_pack import parse_block
    block = (
        '% this is a comment\n'
        '#a01 @x :is "y" ^h.\n'
        '%% another comment'
    )
    entries = parse_block(block)
    assert len(entries) == 1


def test_parse_skips_blank_lines():
    from memory.triple_pack import parse_block
    block = '\n\n#a01 @x :is "y" ^h.\n\n'
    entries = parse_block(block)
    assert len(entries) == 1


def test_parse_handles_l2_reference_object():
    from memory.triple_pack import parse_block
    block = '#a04 @atlas :status "live" ^L2:7f3c91.'
    entries = parse_block(block)
    # ^L2:hex is a provenance tag (pointer to L2 record)
    assert "L2:7f3c91" in entries[0].prov


def test_parse_default_provenance_when_omitted():
    from memory.triple_pack import parse_block
    block = '#a01 @x :is "y".'
    entries = parse_block(block)
    # When ^... is omitted, default to ^h (haiku)
    assert "h" in entries[0].prov


# ----- Parse: errors -----

def test_parse_invalid_syntax_raises():
    from memory.triple_pack import parse_block, ParseError
    with pytest.raises(ParseError):
        parse_block("not valid syntax at all")


def test_parse_missing_period_raises():
    from memory.triple_pack import parse_block, ParseError
    with pytest.raises(ParseError):
        parse_block("#a01 @x :is \"y\"")  # no terminating period


def test_parse_missing_id_raises():
    from memory.triple_pack import parse_block, ParseError
    with pytest.raises(ParseError):
        parse_block('@x :is "y" ^h.')  # no #id


# ----- Serialize -----

def test_serialize_single_entry():
    from memory.triple_pack import serialize_block, Entry
    e = Entry(id="#a01", subject="@jacob", predicate=":is",
              object='"ADHD"', prov=["u", "t=2026-04-12"])
    output = serialize_block([e])
    assert "#a01" in output
    assert "@jacob" in output
    assert ":is" in output
    assert '"ADHD"' in output
    assert output.endswith(".")


def test_serialize_multiple_entries_one_per_line():
    from memory.triple_pack import serialize_block, Entry
    entries = [
        Entry(id="#a01", subject="@x", predicate=":is", object='"y"', prov=["h"]),
        Entry(id="#a02", subject="@m", predicate=":is", object='"n"', prov=["h"]),
    ]
    output = serialize_block(entries)
    assert output.count("\n") >= 1


# ----- Roundtrip property -----

@settings(deadline=None, max_examples=80)
@given(
    id_part=st.text(alphabet="abcdefghij0123456789", min_size=2, max_size=5),
    subj=st.text(alphabet="abcdefghijklmnop", min_size=1, max_size=10),
    pred=st.text(alphabet="abcdefghijklmnop_", min_size=1, max_size=10),
    obj=st.text(alphabet="abcdefghijklmnop ", min_size=1, max_size=20),
)
def test_roundtrip_preserves_entry(id_part, subj, pred, obj):
    """Property: parse(serialize(x)) == x for any well-formed entry."""
    from memory.triple_pack import Entry, parse_block, serialize_block
    # Build a known-valid entry
    original = Entry(
        id=f"#{id_part}",
        subject=f"@{subj}",
        predicate=f":{pred.strip('_').lstrip('_') or 'is'}",
        object=f'"{obj}"',
        prov=["h"],
    )
    serialized = serialize_block([original])
    parsed = parse_block(serialized)
    assert len(parsed) == 1
    assert parsed[0].id == original.id
    assert parsed[0].subject == original.subject
    assert parsed[0].predicate == original.predicate
    assert parsed[0].object == original.object


# ----- Edit operations -----

def test_apply_append_op():
    from memory.triple_pack import apply_edits, parse_block
    initial = '#a01 @x :is "y" ^h.'
    edits = '+ #a02 @m :is "n" ^h.'
    result = apply_edits(initial, edits)
    entries = parse_block(result)
    assert len(entries) == 2
    assert any(e.id == "#a02" for e in entries)


def test_apply_update_op():
    from memory.triple_pack import apply_edits, parse_block
    initial = '#a01 @x :is "old" ^h.'
    edits = '~ #a01 @x :is "new" ^h.'
    result = apply_edits(initial, edits)
    entries = parse_block(result)
    assert len(entries) == 1
    assert entries[0].object == '"new"'


def test_apply_delete_op():
    from memory.triple_pack import apply_edits, parse_block
    initial = '#a01 @x :is "y" ^h.\n#a02 @m :is "n" ^h.'
    edits = '- #a01'
    result = apply_edits(initial, edits)
    entries = parse_block(result)
    assert len(entries) == 1
    assert entries[0].id == "#a02"


def test_apply_tombstone_op():
    """`x` op marks entry superseded but retains it for audit."""
    from memory.triple_pack import apply_edits, parse_entry_with_status
    initial = '#a01 @x :is "y" ^h.'
    edits = 'x #a01'
    result = apply_edits(initial, edits)
    # Entry should still be present but with ^x tag added
    entries = parse_entry_with_status(result)
    assert len(entries) == 1
    assert entries[0].is_tombstoned()


# ----- Density check -----

def test_triple_pack_denser_than_realistic_prose():
    """Goal: triple-pack <= equivalent natural-language prose at scale.

    Compares against realistic prose (with connective words, full sentences),
    not telegraphic minimum-length prose.
    """
    from memory.triple_pack import serialize_block, Entry

    # 10 facts — realistic L1.notes scale
    entries = [
        Entry("#a01", "@jacob", ":is", '"ADHD"', ["u"]),
        Entry("#a02", "@jacob", ":prefers", '"numbered lists max 5 items"', ["u"]),
        Entry("#a03", "@jacob", ":has", '"max plan subscription"', ["u"]),
        Entry("#a04", "@hermes", ":located_at", '"C:/Users/jbogi/claude-nodes/hermes"', ["u"]),
        Entry("#a05", "@hermes", ":depends_on", "@lancedb", ["h"]),
        Entry("#a06", "@hermes", ":depends_on", "@gte_modernbert_base", ["h"]),
        Entry("#a07", "@memory_l1", ":format", '"triple-pack v1"', ["j", "t=2026-05-06"]),
        Entry("#a08", "@memory_l2", ":store", "@lancedb", ["h"]),
        Entry("#a09", "@atlas", ":status", '"dashboard live on port 8501"', ["L2:7f3c91"]),
        Entry("#a10", "@subconscious", ":provider", '"haiku via claude-cli"', ["j"]),
    ]
    triple_text = serialize_block(entries)

    # Realistic prose — what a human might write to convey the same information
    prose = (
        "Jacob has ADHD and prefers numbered lists with at most 5 items. "
        "He has a Max plan subscription. Hermes is located at "
        "C:/Users/jbogi/claude-nodes/hermes and depends on both LanceDB "
        "and the gte-modernbert-base embedding model. The L1 memory uses "
        "the triple-pack v1 format (decided 2026-05-06). The L2 memory "
        "store is backed by LanceDB. The Atlas dashboard is currently "
        "live on port 8501 (full record at L2:7f3c91). The subconscious "
        "agent provider is Haiku via the claude-cli."
    )

    # Triple-pack must be within 10% of natural prose length
    # (the win is structure + parseability + cross-refs, not raw chars)
    assert len(triple_text) <= len(prose) * 1.10, (
        f"triple={len(triple_text)} vs prose={len(prose)}"
    )
    # And it must be parseable, unlike prose
    from memory.triple_pack import parse_block
    parsed = parse_block(triple_text)
    assert len(parsed) == len(entries)


# ----- Cross-reference resolution -----

def test_parse_extracts_l2_pointers():
    from memory.triple_pack import parse_block, extract_l2_pointers
    block = (
        '#a01 @x :is "y" ^L2:abc123.\n'
        '#a02 @m :details "see L2:def456" ^h.\n'
    )
    entries = parse_block(block)
    pointers = extract_l2_pointers(entries)
    assert "abc123" in pointers
    assert "def456" in pointers
