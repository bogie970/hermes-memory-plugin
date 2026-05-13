"""Thin bridge: spawned by TS hooks, imports and runs the Python worker from hermes.

The TS hook sets HERMES_ROOT env var before spawning this script.
This script adds HERMES_ROOT to sys.path and delegates to the actual runner.

Usage: python local_worker.py <payload.json>
"""

import os
import sys

hermes_root = os.environ.get("HERMES_ROOT")
if not hermes_root:
    print("ERROR: HERMES_ROOT environment variable not set", file=sys.stderr)
    sys.exit(1)

# Insert hermes/aisys FIRST so `memory.*` resolves to the canonical hermes copy.
# Otherwise PYTHONPATH (which points to claude-subconscious/python/) wins and
# we end up importing a stale duplicate of the memory package.
sys.path.insert(0, os.path.join(hermes_root, "aisys"))
sys.path.insert(1, hermes_root)

from aisys.subconscious.runner import main  # noqa: E402

main(sys.argv[1] if len(sys.argv) > 1 else None)
