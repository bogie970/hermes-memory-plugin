"""Pydantic models for the memory system.

Memory types follow the CoALA taxonomy (Sumers et al., TMLR 2024):
- episodic: specific past events/sessions
- semantic: generalized facts/rules/insights
- procedural: how-to workflows and decision templates
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _parse_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    memory_type: MemoryType
    category: str  # pricing | incident | expansion | market | workflow | rule | session | system
    source: str = ""  # who wrote it: hermes | atlas | auto_pricer | collector | manual | migration
    namespace: str = ""  # visibility scope: defaults to source if empty
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)  # IDs of related memories (A-MEM pattern)
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    archived: bool = False

    def enriched_text(self) -> str:
        """Return enriched text for embedding if enrichment data exists."""
        enrichment = self.metadata.get("enrichment", {})
        if not enrichment:
            return self.content
        from memory.enrichment import build_enriched_text
        return build_enriched_text(self.content, enrichment)

    def to_lance_dict(self, vector: list[float], include_v2: bool = True) -> dict[str, Any]:
        """Convert to a flat dict for LanceDB insertion.

        If include_v2=True (default), populates v2 schema fields with sensible
        defaults. Set False only when targeting a v1 (pre-migration) table.
        """
        d = {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "category": self.category,
            "source": self.source,
            "namespace": self.namespace or self.source,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "tags": ",".join(self.tags),
            "links": ",".join(self.links),
            "session_id": self.session_id or "",
            "metadata_json": self.model_dump_json(include={"metadata"}),
            "archived": self.archived,
            "vector": vector,
        }
        if include_v2:
            now_iso = self.created_at.isoformat()
            v2_defaults = self.metadata.get("_v2_overrides", {})
            d.update({
                "tier": v2_defaults.get("tier", "verified"),
                "provenance": v2_defaults.get("provenance", "llm_inferred"),
                "source_ref": v2_defaults.get("source_ref", "memory_store_insert"),
                "writer": v2_defaults.get("writer", "manual"),
                "confidence": v2_defaults.get("confidence", 0.5),
                "valid_from": v2_defaults.get("valid_from", now_iso),
                "valid_to": v2_defaults.get("valid_to", ""),
                "supersedes": v2_defaults.get("supersedes", ""),
                "superseded_by": v2_defaults.get("superseded_by", ""),
                "contradiction_state": v2_defaults.get("contradiction_state", "clean"),
                "conflict_with": v2_defaults.get("conflict_with", "[]"),
                "seen_count": v2_defaults.get("seen_count", 1),
                "last_seen_at": v2_defaults.get("last_seen_at", now_iso),
                "promoted_at": v2_defaults.get("promoted_at", ""),
                "embedding_model": v2_defaults.get("embedding_model", "gte-modernbert-base"),
                "embedding_version": v2_defaults.get("embedding_version", 1),
            })
        return d

    @classmethod
    def from_lance_row(cls, row: dict[str, Any]) -> MemoryRecord:
        """Reconstruct from a LanceDB row dict."""
        import json

        tags = [t for t in row.get("tags", "").split(",") if t]
        links = [l for l in row.get("links", "").split(",") if l]
        meta_raw = row.get("metadata_json", '{"metadata":{}}')
        try:
            metadata = json.loads(meta_raw).get("metadata", {})
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return cls(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            category=row["category"],
            source=row["source"],
            namespace=row.get("namespace", row.get("source", "")),
            created_at=_parse_datetime(row["created_at"]),
            last_accessed=_parse_datetime(row["last_accessed"]),
            access_count=row.get("access_count", 0),
            importance=row.get("importance", 0.5),
            tags=tags,
            links=links,
            session_id=row.get("session_id") or None,
            metadata=metadata,
            archived=row.get("archived", False),
        )


class ScoredMemory(BaseModel):
    """A memory record with retrieval scores attached."""

    record: MemoryRecord
    relevance: float = 0.0
    recency: float = 0.0
    importance_score: float = 0.0
    combined_score: float = 0.0
    vector_distance: float = 0.0


class ProcedureStep(BaseModel):
    action: str  # run_command | check_condition | wait | decide | note
    content: str
    expected_output: str | None = None
    fallback: str | None = None


class Procedure(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    steps: list[ProcedureStep] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    last_used: datetime | None = None
    source_sessions: list[str] = Field(default_factory=list)
    status: str = "candidate"  # candidate | confirmed | deprecated
