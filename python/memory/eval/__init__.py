"""Memory eval harness — recall@k regression detection.

Run: python -m memory.eval.harness --fixtures fixtures.yaml
"""

from memory.eval.harness import Fixture, Report, FixtureRow, run, load_fixtures
from memory.eval.synthesize import synthesize

__all__ = ["Fixture", "Report", "FixtureRow", "run", "load_fixtures", "synthesize"]
