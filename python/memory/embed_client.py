"""Client for the embedding daemon.

Usage:
    client = DaemonClient()
    vectors = client.embed(["text 1", "text 2"])

Tries to connect to the running daemon; on connection failure, spawns
one detached and retries. On persistent failure, raises DaemonUnavailable
so the caller can fall back to in-process embedding.

The daemon is single-instance per machine (spawn protected by FileLock).
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
import uuid
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from memory.embed_transport import (
    connect,
    default_socket_path,
    recv_msg,
    runtime_dir,
    send_msg,
)


SPAWN_RETRY_DELAYS = [0.2, 0.4, 0.8, 1.5, 3.0, 5.0, 8.0]  # ~19s total
PING_TIMEOUT = 2.0


class DaemonUnavailable(Exception):
    """Raised when the daemon cannot be reached and cannot be spawned."""


def _pid_alive(pid: int) -> bool:
    """Cross-platform: is process pid still running?"""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            # On Windows, os.kill with signal 0 doesn't work the same way;
            # use OpenProcess via ctypes.
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if handle == 0:
                return False
            kernel32.CloseHandle(handle)
            return True
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError):
        return False


class DaemonClient:
    def __init__(self, socket_path: pathlib.Path | None = None):
        self.socket_path = socket_path or default_socket_path()
        self.pid_path = runtime_dir() / "embed_daemon.pid"

    # ----- daemon discovery / spawn -----

    def _daemon_pid(self) -> int | None:
        if not self.pid_path.exists():
            return None
        try:
            return int(self.pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def _daemon_alive(self) -> bool:
        pid = self._daemon_pid()
        if pid is None:
            return False
        if not _pid_alive(pid):
            # Stale PID file — clean up
            try:
                self.pid_path.unlink()
            except OSError:
                pass
            return False
        # Try a ping
        try:
            with self._connect(timeout=PING_TIMEOUT) as conn:
                send_msg(conn, {"op": "ping"})
                reply = recv_msg(conn)
                return bool(reply and reply.get("ok"))
        except (ConnectionError, OSError, ValueError):
            return False

    def _spawn_daemon(self) -> None:
        """Spawn the daemon detached. Protected by a spawn-time file lock
        so two clients don't double-spawn."""
        from filelock import FileLock, Timeout
        lock_path = runtime_dir() / "embed_daemon.spawn.lock"
        try:
            with FileLock(str(lock_path), timeout=1.0):
                # Re-check after acquiring lock
                if self._daemon_alive():
                    return
                self._do_spawn()
        except Timeout:
            # Another client is spawning; just wait briefly
            time.sleep(1.0)

    def _do_spawn(self) -> None:
        """Detached subprocess spawn — survives parent's exit."""
        cmd = [sys.executable, "-m", "memory.embed_daemon"]
        env = dict(os.environ)
        # Set PYTHONPATH so the daemon can import its own modules when
        # spawned from arbitrary cwd.
        # Find the parent of the memory/ package (the dir containing the
        # memory namespace)
        memory_parent = str(pathlib.Path(__file__).resolve().parent.parent)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{memory_parent}{os.pathsep}{existing}" if existing else memory_parent
        )

        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": env,
            "close_fds": True,
        }
        if sys.platform == "win32":
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, hidden
            kwargs["creationflags"] = 0x00000008 | 0x00000200
        else:
            kwargs["start_new_session"] = True

        subprocess.Popen(cmd, **kwargs)

    def _connect(self, timeout: float = 2.0):
        return connect(self.socket_path, timeout=timeout)

    def _ensure_running(self) -> None:
        """Connect, spawning the daemon if needed. Raises DaemonUnavailable
        if we can't make contact within the retry budget."""
        if self._daemon_alive():
            return
        # Spawn
        self._spawn_daemon()
        # Wait for it to be reachable
        for delay in SPAWN_RETRY_DELAYS:
            time.sleep(delay)
            if self._daemon_alive():
                return
        raise DaemonUnavailable(
            f"daemon not reachable at {self.socket_path} after "
            f"{sum(SPAWN_RETRY_DELAYS):.1f}s"
        )

    # ----- public API -----

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns 768-dim vectors.

        Raises DaemonUnavailable on persistent connection failure.
        """
        if not texts:
            return []
        self._ensure_running()
        req_id = uuid.uuid4().hex[:8]
        try:
            with self._connect(timeout=60.0) as conn:
                send_msg(conn, {"op": "embed", "texts": texts, "id": req_id})
                reply = recv_msg(conn)
        except (ConnectionError, OSError, ValueError) as e:
            raise DaemonUnavailable(f"embed call failed: {e}") from e

        if reply is None:
            raise DaemonUnavailable("daemon closed connection")
        if "error" in reply:
            raise DaemonUnavailable(f"daemon error: {reply.get('error')}")
        vectors = reply.get("vectors", [])
        if not isinstance(vectors, list):
            raise DaemonUnavailable("malformed daemon reply")
        return vectors

    def embed_one(self, text: str) -> list[float]:
        result = self.embed([text])
        return result[0] if result else []

    def ping(self) -> dict | None:
        """Return daemon status, or None if unreachable."""
        try:
            with self._connect(timeout=PING_TIMEOUT) as conn:
                send_msg(conn, {"op": "ping"})
                return recv_msg(conn)
        except (ConnectionError, OSError, ValueError):
            return None

    def shutdown(self) -> bool:
        """Ask the daemon to exit. Returns True if it acknowledged."""
        pid = self._daemon_pid()
        if pid is None:
            return False
        try:
            with self._connect(timeout=PING_TIMEOUT) as conn:
                send_msg(conn, {"op": "shutdown", "token": str(pid)})
                reply = recv_msg(conn)
                return bool(reply and reply.get("ok"))
        except (ConnectionError, OSError, ValueError):
            return False
