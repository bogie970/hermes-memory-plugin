"""Phase E — eval harness tests.

Recall@k regression detection: catch retrieval drift over time.
Fixture synthesis: generate (query, expected_id) pairs via LLM.

Run: pytest aisys/memory/tests/test_eval_harness.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


@pytest.fixture
def v2_store(lance_store_v1):
    from memory.migrate_v2 import backfill_v2
    backfill_v2(lance_store_v1)
    return lance_store_v1


@pytest.fixture
def seeded_store(v2_store):
    """Store seeded with known content + known ids for recall testing."""
    from memory.write_gate import write_memory
    seeds = [
        ("hermes is the AI co-pilot for Jacob's physics work", "physics"),
        ("LanceDB is the chosen vector store for hermes memory", "lancedb"),
        ("Atlas dashboard runs on streamlit at port 8501", "atlas"),
        ("subconscious worker uses Haiku via claude --print", "subconscious"),
        ("triple-pack format makes L1 dense and parseable", "format"),
    ]
    ids = {}
    for content, key in seeds:
        rec_id = write_memory(
            store=v2_store,
            content=content,
            writer="user",
            provenance="user_stated",
            source_ref=f"seed:{key}",
            confidence=1.0,
        )
        ids[key] = rec_id
    return v2_store, ids


# ----- Fixture loading + format -----

def test_load_fixtures_from_yaml(tmp_path):
    from memory.eval.harness import load_fixtures
    fixture_path = tmp_path / "fixtures.yaml"
    fixture_path.write_text("""\
- id: f1
  query: what is the AI co-pilot's name?
  expected_memory_ids: [m1, m2]
  k: 5
  category: single_hop
- id: f2
  query: irrelevant noise
  expected_memory_ids: []
  category: negative
""")
    fixtures = load_fixtures(str(fixture_path))
    assert len(fixtures) == 2
    assert fixtures[0].id == "f1"
    assert fixtures[0].category == "single_hop"
    assert fixtures[1].category == "negative"


def test_fixture_default_k_is_5(tmp_path):
    from memory.eval.harness import load_fixtures
    fixture_path = tmp_path / "fixtures.yaml"
    fixture_path.write_text("""\
- id: f1
  query: x
  expected_memory_ids: [a]
  category: single_hop
""")
    fixtures = load_fixtures(str(fixture_path))
    assert fixtures[0].k == 5


# ----- run() — recall@k computation -----

def test_run_computes_recall_for_perfect_match(seeded_store, tmp_path):
    """Query embeds to exact same vector as a seed -> recall@5 == 1.0."""
    store, ids = seeded_store
    from memory.eval.harness import Fixture, run
    fixtures = [
        Fixture(id="f1",
                query="hermes is the AI co-pilot for Jacob's physics work",
                expected_memory_ids=[ids["physics"]],
                k=5, category="single_hop"),
    ]
    report = run(fixtures, store)
    assert report.mean_recall_at_k == pytest.approx(1.0)


def test_run_handles_negative_fixtures(seeded_store):
    """Negative tests pass when scores are below threshold OR no expected ids match."""
    store, _ids = seeded_store
    from memory.eval.harness import Fixture, run
    fixtures = [
        Fixture(id="neg1",
                query="totally unrelated content about quantum chromodynamics",
                expected_memory_ids=[],
                k=5, category="negative"),
    ]
    report = run(fixtures, store)
    # Negative test passes if no expected ids matched (which is true since list is empty)
    neg_rows = [r for r in report.rows if r.fixture.category == "negative"]
    assert len(neg_rows) == 1
    assert neg_rows[0].passed


def test_run_reports_per_fixture_metrics(seeded_store):
    """Report has per-fixture rows with recall, mrr, scores."""
    store, ids = seeded_store
    from memory.eval.harness import Fixture, run
    fixtures = [
        Fixture(id="f1", query="hermes physics co-pilot",
                expected_memory_ids=[ids["physics"]],
                k=5, category="single_hop"),
        Fixture(id="f2", query="LanceDB vector store",
                expected_memory_ids=[ids["lancedb"]],
                k=5, category="single_hop"),
    ]
    report = run(fixtures, store)
    assert len(report.rows) == 2
    assert all(0.0 <= r.recall_at_k <= 1.0 for r in report.rows)
    assert all(0.0 <= r.mrr <= 1.0 for r in report.rows)


def test_run_appends_to_history(seeded_store, tmp_path):
    """Running with history_path appends a parquet/JSON row."""
    store, ids = seeded_store
    from memory.eval.harness import Fixture, run
    history_path = tmp_path / "eval_history.jsonl"
    fixtures = [
        Fixture(id="f1", query="LanceDB", expected_memory_ids=[ids["lancedb"]],
                k=5, category="single_hop"),
    ]
    run(fixtures, store, history_path=str(history_path))
    assert history_path.exists()
    lines = history_path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert "mean_recall_at_k" in entry
    assert "run_ts" in entry
    assert "corpus_size" in entry


# ----- synthesize() — generate fixtures from existing memories -----

def test_synthesize_produces_fixtures_from_seeds(seeded_store):
    """LLM generates 1 retrieval query per seed + 1 paraphrase + 1 negative."""
    store, _ids = seeded_store

    def fake_llm(messages, **kw):
        # Return a JSON list per seed
        return {"role": "assistant", "content": json.dumps([
            {"query": "what is the AI co-pilot?",
             "kind": "direct"},
            {"query": "tell me about Jacob's assistant",
             "kind": "paraphrase"},
            {"query": "list of all rare birds in Antarctica",
             "kind": "negative"},
        ])}

    from memory.eval.synthesize import synthesize
    fixtures = synthesize(store, n_seeds=1, chat_fn=fake_llm)
    # Each seed yields 3 fixtures (direct + paraphrase + negative)
    assert len(fixtures) >= 3
    # At least one should have the negative category
    assert any(f.category == "negative" for f in fixtures)
