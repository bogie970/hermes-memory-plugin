"""Entry point for the subconscious background worker.

Called from the TS hook via: python -m subconscious.runner <payload.json>

Follows the same interface contract as the original send_worker_sdk.ts:
1. Read payload JSON from CLI arg (file path)
2. Process transcript through the agentic loop
3. Update state file with lastProcessedIndex
4. Delete payload file
5. Exit 0 on success, 1 on failure
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger("subconscious.runner")


def _setup_logging(log_dir: str) -> None:
    """Set up file logging to match the TS worker's log pattern."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "local_worker.log")

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def _get_temp_state_dir() -> str:
    """Match the TS getTempStateDir() logic — stable dir per user."""
    tmp = os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp"
    import getpass
    user = getpass.getuser()
    return os.path.join(tmp, f"letta-claude-sync-{user}")


def _get_blocks_path(cwd: str) -> str:
    """Resolve blocks JSON path, matching local_store.ts logic."""
    home = os.environ.get("LETTA_HOME", cwd)
    if home.startswith("~"):
        home = os.path.expanduser(home)
    return os.path.join(home, ".letta", "claude", "local_blocks.json")


def _check_singleton(cwd: str) -> bool:
    """Check if another worker is already running for this project.

    Uses a PID file. Returns True if we can proceed, False if another
    worker is active.
    """
    home = os.environ.get("LETTA_HOME", cwd)
    pid_file = os.path.join(home, ".letta", "claude", "worker.pid")

    if os.path.exists(pid_file):
        try:
            content = Path(pid_file).read_text(encoding="utf-8").strip()
            parts = content.split(":")
            old_pid = int(parts[0])
            old_time = float(parts[1]) if len(parts) > 1 else 0
            from .config import LOOP_TIMEOUT_SECONDS
            stale_threshold = LOOP_TIMEOUT_SECONDS + 30
            if time.time() - old_time > stale_threshold:
                log.info("PID file is stale (%.0fs old) — proceeding", time.time() - old_time)
            elif sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                handle = kernel32.OpenProcess(0x0400, False, old_pid)  # PROCESS_QUERY_INFORMATION
                if handle:
                    exit_code = ctypes.c_ulong()
                    kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                    kernel32.CloseHandle(handle)
                    if exit_code.value == 259:  # STILL_ACTIVE
                        log.info("Another worker (PID %d) is still running — skipping", old_pid)
                        return False
            else:
                os.kill(old_pid, 0)
                log.info("Another worker (PID %d) is still running — skipping", old_pid)
                return False
        except (ValueError, OSError, ProcessLookupError):
            pass  # stale PID file, proceed

    # Write our PID
    os.makedirs(os.path.dirname(pid_file), exist_ok=True)
    Path(pid_file).write_text(f"{os.getpid()}:{time.time():.0f}", encoding="utf-8")
    return True


def _ollama_process_exists() -> bool:
    """Check if an ollama process is already running (even if not yet responding)."""
    import subprocess

    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq ollama.exe", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return "ollama.exe" in result.stdout
        else:
            result = subprocess.run(
                ["pgrep", "-x", "ollama"],
                capture_output=True, timeout=5,
            )
            return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _start_ollama() -> bool:
    """Start Ollama if no instance exists. Returns True if it comes up."""
    import shutil
    import subprocess

    from .config import OLLAMA_HOST
    from . import ollama_chat

    if _ollama_process_exists():
        log.info("Ollama process already exists — waiting for it to become ready")
        for _ in range(15):
            time.sleep(1)
            if ollama_chat.ping(OLLAMA_HOST, timeout=2):
                log.info("Existing Ollama instance is now responding")
                return True
        log.warning("Ollama process exists but not responding after 15s")
        return False

    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        log.warning("ollama binary not found in PATH")
        return False

    log.info("No Ollama process found — starting one")
    try:
        if sys.platform == "win32":
            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
            )
        else:
            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    except OSError as e:
        log.warning("Failed to spawn ollama serve: %s", e)
        return False

    for _ in range(15):
        time.sleep(1)
        if ollama_chat.ping(OLLAMA_HOST, timeout=2):
            log.info("Ollama started successfully")
            return True

    log.warning("Ollama did not come up after 15s")
    return False


def _ensure_model(host: str, model: str) -> bool:
    """Check if model is available locally; pull it if not."""
    import requests

    try:
        r = requests.post(
            f"{host.rstrip('/')}/api/show",
            json={"name": model},
            timeout=5,
        )
        if r.status_code == 200:
            return True
    except requests.RequestException:
        return False

    log.info("Model %s not found locally — pulling (this may take a while)", model)
    try:
        r = requests.post(
            f"{host.rstrip('/')}/api/pull",
            json={"name": model, "stream": False},
            timeout=600,
        )
        if r.status_code == 200:
            log.info("Model %s pulled successfully", model)
            return True
        log.warning("Model pull failed: %s %s", r.status_code, r.text[:200])
    except requests.RequestException as e:
        log.warning("Model pull request failed: %s", e)

    return False


def _cleanup_pid(cwd: str) -> None:
    home = os.environ.get("LETTA_HOME", cwd)
    pid_file = os.path.join(home, ".letta", "claude", "worker.pid")
    try:
        if os.path.exists(pid_file):
            content = Path(pid_file).read_text(encoding="utf-8").strip()
            stored_pid = content.split(":")[0]
            if stored_pid == str(os.getpid()):
                os.unlink(pid_file)
    except OSError:
        pass


def main(payload_path: str | None = None) -> None:
    """Main entry point. Reads payload, runs agentic loop, updates state."""
    if payload_path is None:
        if len(sys.argv) < 2:
            print("Usage: python -m aisys.subconscious.runner <payload.json>", file=sys.stderr)
            sys.exit(1)
        payload_path = sys.argv[1]

    # Respect LETTA_MODE=off kill switch
    if os.environ.get("LETTA_MODE", "").lower() == "off":
        sys.exit(0)

    # Setup logging
    _setup_logging(_get_temp_state_dir())
    log.info("=" * 60)
    log.info("Local worker started, payload: %s", payload_path)

    # Read payload
    try:
        payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.error("Failed to read payload: %s", e)
        sys.exit(1)

    cwd = payload.get("cwd", ".")
    session_id = payload.get("sessionId", "")
    state_file = payload.get("stateFile", "")
    new_last_index = payload.get("newLastProcessedIndex", -1)
    transcript_xml = payload.get("transcriptXml", "")

    # Truncate transcript to prevent OOM on long sessions (keep tail = most recent)
    from .config import TRANSCRIPT_MAX_CHARS
    if len(transcript_xml) > TRANSCRIPT_MAX_CHARS:
        log.info("Truncating transcript from %d to %d chars", len(transcript_xml), TRANSCRIPT_MAX_CHARS)
        transcript_xml = transcript_xml[-TRANSCRIPT_MAX_CHARS:]

    log.info("Session: %s, cwd: %s", session_id, cwd)
    log.info("Transcript length: %d chars", len(transcript_xml))

    # Singleton check
    if not _check_singleton(cwd):
        log.info("Skipping — another worker is active")
        sys.exit(0)

    try:
        # Skip if transcript is too short (no meaningful content)
        if len(transcript_xml) < 50:
            log.info("Transcript too short (%d chars), skipping", len(transcript_xml))
            _update_state(state_file, new_last_index)
            _cleanup_payload(payload_path)
            return

        # Ensure Ollama is running and model is available
        from . import ollama_chat
        from .config import OLLAMA_HOST, OLLAMA_MODEL

        if not ollama_chat.ping(OLLAMA_HOST):
            log.info("Ollama not running — attempting to start")
            if not _start_ollama():
                log.error("Could not start Ollama — skipping")
                _write_status(cwd, error="Ollama could not be started")
                _cleanup_pid(cwd)
                sys.exit(0)

        if not _ensure_model(OLLAMA_HOST, OLLAMA_MODEL):
            log.error("Model %s unavailable and pull failed — skipping", OLLAMA_MODEL)
            _write_status(cwd, error=f"Model {OLLAMA_MODEL} unavailable")
            _cleanup_pid(cwd)
            sys.exit(0)

        # Initialize block store
        blocks_path = _get_blocks_path(cwd)
        from .blocks import BlockStore
        store = BlockStore(blocks_path)

        log.info("Blocks path: %s", blocks_path)
        log.info("Block summary: %s", store.summary())

        # Try to initialize vector memory store for conversation_search
        memory_store = None
        try:
            from memory.store import MemoryStore
            from memory.config import LANCEDB_PATH
            memory_store = MemoryStore(db_path=LANCEDB_PATH)
            log.info("Vector memory store initialized at %s", LANCEDB_PATH)
        except Exception as e:
            log.warning("Vector memory store unavailable: %s", e)

        # Run the agentic loop
        from .loop import run_loop
        result = run_loop(
            store=store,
            transcript_xml=transcript_xml,
            session_id=session_id,
            model=OLLAMA_MODEL,
            host=OLLAMA_HOST,
            memory_store=memory_store,
        )

        log.info(
            "Loop result: %d iterations, %d tool calls, %.1fs, error=%s",
            result.iterations,
            len(result.tool_calls_made),
            result.duration_seconds,
            result.error,
        )

        if result.tool_calls_made:
            for tc in result.tool_calls_made:
                log.info("  Tool: %s(%s) -> %s", tc["name"], list(tc["args"].keys()), tc["result"][:100])

        if result.final_response:
            log.info("Final response: %s", result.final_response[:300])

        # --- Phase 2: Consolidation into L2 vector store (incremental) ---
        try:
            from .consolidation import ConsolidationEngine, parse_transcript_messages
            from memory.llm_caller import get_llm_caller

            # Try claude-cli-standalone first (uses subscription), fall back to ollama
            llm = get_llm_caller("claude-cli-standalone", model="sonnet", timeout=120)
            if llm is None:
                log.info("claude-cli-standalone unavailable, falling back to ollama for consolidation")
                llm = get_llm_caller("ollama", model=OLLAMA_MODEL, timeout=120)

            if llm and memory_store:
                parsed_messages = parse_transcript_messages(transcript_xml)
                if parsed_messages:
                    # Only consolidate messages we haven't processed yet.
                    # new_last_index tracks the payload's newLastProcessedIndex.
                    # First run has lastProcessedIndex = -1, so all messages are new.
                    if new_last_index > 0 and len(parsed_messages) > new_last_index:
                        new_messages = parsed_messages[new_last_index:]
                        log.info(
                            "Incremental consolidation: %d total messages, processing %d new (from index %d)",
                            len(parsed_messages), len(new_messages), new_last_index,
                        )
                    else:
                        new_messages = parsed_messages

                    if new_messages:
                        engine = ConsolidationEngine(memory_store, llm)
                        consol_result = engine.process_batch(new_messages, session_id)
                        log.info(
                            "Consolidation: %d extracted, %d stored, %d merged, %d skipped",
                            consol_result.extracted,
                            consol_result.stored,
                            consol_result.merged,
                            consol_result.skipped,
                        )
                        if consol_result.errors:
                            log.warning("Consolidation errors: %s", consol_result.errors[:3])
                    else:
                        log.info("No new messages to consolidate after slicing")
                else:
                    log.info("No parseable messages in transcript — skipping consolidation")
            else:
                log.info("Consolidation skipped: llm=%s, memory_store=%s",
                         "ok" if llm else "none", "ok" if memory_store else "none")
        except Exception as e:
            log.warning("Consolidation failed (non-fatal): %s", e, exc_info=True)

        # --- Phase 5: Periodic tiering/decay/merge ---
        if memory_store:
            try:
                _run_periodic_maintenance(cwd, memory_store)
            except Exception as e:
                log.warning("Periodic maintenance failed (non-fatal): %s", e, exc_info=True)

        # Update state and cleanup
        _update_state(state_file, new_last_index)
        _cleanup_payload(payload_path)
        _write_status(cwd, success=True, tool_calls=len(result.tool_calls_made))

        log.info("Worker completed successfully")

    except Exception as e:
        log.error("Worker failed: %s", e, exc_info=True)
        _write_status(cwd, error=str(e))
        sys.exit(1)
    finally:
        _cleanup_pid(cwd)


def _get_tiering_timestamp_path(cwd: str) -> str:
    """Path to the file tracking when tiering last ran."""
    home = os.environ.get("LETTA_HOME", cwd)
    return os.path.join(home, ".letta", "claude", "last_tiering_run.txt")


def _run_periodic_maintenance(cwd: str, memory_store: "MemoryStore") -> None:
    """Run tiering/decay/merge if enough time has passed since last run.

    Only runs if it's been more than 1 hour since the last tiering pass.
    Writes a timestamp file after completing.
    """
    ts_path = _get_tiering_timestamp_path(cwd)
    min_interval_seconds = 3600  # 1 hour

    # Check if we should run
    if os.path.exists(ts_path):
        try:
            last_run_str = Path(ts_path).read_text(encoding="utf-8").strip()
            last_run = float(last_run_str)
            elapsed = time.time() - last_run
            if elapsed < min_interval_seconds:
                log.info(
                    "Skipping tiering — last run %.0f minutes ago (threshold: %d min)",
                    elapsed / 60, min_interval_seconds // 60,
                )
                return
        except (ValueError, OSError):
            pass  # Corrupt file, run anyway

    log.info("Starting periodic maintenance (tiering/decay/merge)")

    import pathlib as _pl
    import sys as _sys
    _hermes_root = _pl.Path(__file__).resolve().parent.parent.parent
    _aisys_str = str(_hermes_root / "aisys")
    if _aisys_str not in _sys.path:
        _sys.path.insert(0, _aisys_str)

    # 1. Importance decay
    try:
        from memory.consolidation import run_consolidation
        decay_result = run_consolidation(memory_store)
        log.info(
            "Decay: %d checked, %d decayed, %d exempt",
            decay_result.total_checked, decay_result.decayed, decay_result.skipped_exempt,
        )
    except Exception as e:
        log.warning("Decay failed: %s", e, exc_info=True)

    # 2. Archive/promote tiering
    try:
        from memory.tiering import run_tiering
        tier_result = run_tiering(memory_store)
        log.info(
            "Tiering: %d archived, %d promoted",
            tier_result.archived, tier_result.promoted,
        )
    except Exception as e:
        log.warning("Tiering failed: %s", e, exc_info=True)

    # 3. Merge pass
    try:
        from memory.tiering import run_merge_pass
        merge_result = run_merge_pass(memory_store)
        log.info(
            "Merge: %d merged, %d linked (of %d checked)",
            merge_result.merged, merge_result.linked, merge_result.total_checked,
        )
    except Exception as e:
        log.warning("Merge failed: %s", e, exc_info=True)

    # Write timestamp
    try:
        os.makedirs(os.path.dirname(ts_path), exist_ok=True)
        Path(ts_path).write_text(f"{time.time():.0f}", encoding="utf-8")
        log.info("Periodic maintenance complete — timestamp written")
    except OSError as e:
        log.warning("Failed to write tiering timestamp: %s", e)


def _update_state(state_file: str, new_last_index: int) -> None:
    """Update the sync state file with new lastProcessedIndex. Atomic write."""
    if not state_file:
        return
    try:
        if os.path.exists(state_file):
            state = json.loads(Path(state_file).read_text(encoding="utf-8"))
        else:
            state_dir = os.path.dirname(state_file)
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)
            state = {}
        state["lastProcessedIndex"] = new_last_index
        import tempfile
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(state_file) or ".",
            prefix=".state_", suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, state_file)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        log.info("Updated state: lastProcessedIndex=%d", new_last_index)
    except (json.JSONDecodeError, OSError) as e:
        log.error("Failed to update state: %s", e)


def _cleanup_payload(payload_path: str) -> None:
    """Delete the payload file after processing."""
    try:
        os.unlink(payload_path)
    except OSError:
        pass


def _write_status(
    cwd: str,
    success: bool = False,
    error: str | None = None,
    tool_calls: int = 0,
) -> None:
    """Write a status file so the sync hooks can check worker health."""
    home = os.environ.get("LETTA_HOME", cwd)
    status_path = os.path.join(home, ".letta", "claude", "worker_status.json")
    status = {
        "last_run": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "success": success,
        "error": error,
        "tool_calls": tool_calls,
        "pid": os.getpid(),
    }
    try:
        os.makedirs(os.path.dirname(status_path), exist_ok=True)
        Path(status_path).write_text(json.dumps(status, indent=2), encoding="utf-8")
    except OSError:
        pass


if __name__ == "__main__":
    main()
