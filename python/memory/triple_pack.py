"""Triple-pack format — dense LLM-readable notation for L1.notes.

Grammar (BNF-ish):
    block      ::= entry ( "\n" entry )*
    entry      ::= id WS subject WS predicate WS object provenance? "."
    id         ::= "#" [a-z0-9]{2,8}
    subject    ::= ref | literal
    predicate  ::= ":" [a-z_]+
    object     ::= ref | literal
    ref        ::= "@" name | "L2:" hex | "#" id_ref
    literal    ::= '"' [^"]* '"' | digits | bool
    provenance ::= " ^" tag ( "," tag )*
    tag        ::= "u" | "h" | "j" | "o" | "x"
                 | "t=" iso_date | "c=" float | "L2:" hex

Edit ops (one per line):
    + <entry>      append new entry
    ~ <entry>      update entry by id
    - #<id>        delete entry
    x #<id>        tombstone (keep, mark superseded)

Comments: lines starting with `%` are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---- Data model ----

@dataclass
class Entry:
    id: str
    subject: str
    predicate: str
    object: str
    prov: list[str] = field(default_factory=list)

    def is_tombstoned(self) -> bool:
        return "x" in self.prov


class ParseError(Exception):
    """Raised when a triple-pack block cannot be parsed."""


# ---- Regex ----

# An entry is: #id @subj :pred OBJ ^prov.
# OBJ can be: "literal", @ref, L2:hex, #idref, number, bare-word
ENTRY_RE = re.compile(
    r"""^
    (\#[a-z0-9]{2,8})\s+               # group 1: id (#xxx)
    (\@[\w/.\-]+|L2:[a-f0-9]+|\#[a-z0-9]{2,8}|"[^"]*"|[\w\-./]+)  # group 2: subject
    \s+
    (:[a-z_]+)\s+                      # group 3: predicate (:verb)
    (\@[\w/.\-]+|L2:[a-f0-9]+|\#[a-z0-9]{2,8}|"[^"]*"|[\w\-./]+|\[[^\]]*\])  # group 4: object
    (?:\s+\^([\w,=.\-:/+]+))?          # group 5: optional provenance
    \s*\.\s*$
    """,
    re.VERBOSE,
)

EDIT_LINE_RE = re.compile(r"^([+~\-x])\s+(.+)$")
ID_ONLY_RE = re.compile(r"^\#[a-z0-9]{2,8}$")


# ---- Parsing ----

def _parse_provenance(prov_str: str | None) -> list[str]:
    if not prov_str:
        return ["h"]  # default: written by haiku
    return [tag.strip() for tag in prov_str.split(",") if tag.strip()]


def parse_block(block: str) -> list[Entry]:
    """Parse a triple-pack block into a list of Entry objects.

    Raises ParseError on the first invalid line. Skips comments and blank lines.
    """
    if not block or not block.strip():
        return []

    entries: list[Entry] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        match = ENTRY_RE.match(line)
        if not match:
            raise ParseError(f"invalid entry: {line!r}")
        id_, subj, pred, obj, prov = match.groups()
        entries.append(Entry(
            id=id_,
            subject=subj,
            predicate=pred,
            object=obj,
            prov=_parse_provenance(prov),
        ))
    return entries


def parse_entry_with_status(block: str) -> list[Entry]:
    """Same as parse_block but does not filter tombstoned entries.

    Tombstones are entries with the `x` provenance tag — kept for audit
    trail but normally hidden from active retrieval.
    """
    return parse_block(block)


# ---- Serializing ----

def _serialize_entry(e: Entry) -> str:
    prov = "^" + ",".join(e.prov) if e.prov else ""
    parts = [e.id, e.subject, e.predicate, e.object]
    line = " ".join(parts)
    if prov:
        line = f"{line} {prov}"
    return f"{line}."


def serialize_block(entries: list[Entry]) -> str:
    """Serialize a list of entries to a triple-pack block string."""
    return "\n".join(_serialize_entry(e) for e in entries)


# ---- Edit operations ----

def apply_edits(initial: str, edit_script: str) -> str:
    """Apply an edit script to a block, returning the new block.

    Edit script lines:
        + <entry>      append new entry
        ~ <entry>      replace entry by id
        - #<id>        delete entry
        x #<id>        mark entry as tombstoned (^x added)
    """
    entries = parse_block(initial)
    by_id = {e.id: e for e in entries}
    order: list[str] = [e.id for e in entries]

    for raw_line in edit_script.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue

        match = EDIT_LINE_RE.match(line)
        if not match:
            raise ParseError(f"invalid edit line: {line!r}")
        op, rest = match.groups()

        if op == "+":
            new_entries = parse_block(rest)
            for e in new_entries:
                if e.id in by_id:
                    continue  # idempotent: skip if id already present
                by_id[e.id] = e
                order.append(e.id)

        elif op == "~":
            new_entries = parse_block(rest)
            for e in new_entries:
                if e.id not in by_id:
                    raise ParseError(f"~ update target not found: {e.id}")
                by_id[e.id] = e

        elif op == "-":
            if not ID_ONLY_RE.match(rest):
                raise ParseError(f"- delete expects bare #id: {rest!r}")
            if rest in by_id:
                del by_id[rest]
                order.remove(rest)

        elif op == "x":
            if not ID_ONLY_RE.match(rest):
                raise ParseError(f"x tombstone expects bare #id: {rest!r}")
            if rest in by_id and "x" not in by_id[rest].prov:
                by_id[rest].prov.append("x")

    return serialize_block([by_id[i] for i in order])


# ---- Cross-reference helpers ----

L2_REF_RE = re.compile(r"L2:([a-f0-9]+)")


def extract_l2_pointers(entries: list[Entry]) -> set[str]:
    """Return the set of L2 record ids referenced anywhere in the entries.

    Looks in both object positions and provenance tags.
    """
    pointers: set[str] = set()
    for e in entries:
        for hit in L2_REF_RE.findall(e.object):
            pointers.add(hit)
        for tag in e.prov:
            for hit in L2_REF_RE.findall(tag):
                pointers.add(hit)
    return pointers
