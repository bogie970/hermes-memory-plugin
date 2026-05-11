"""Embedding daemon — holds one warm SentenceTransformer for the whole machine.

Eliminates the ~25-second cold-load cost per CLI invocation. CLIs that
need embeddings connect via embed_transport; the daemon answers with the
encoded vector. Auto-shuts-down after HERMES_EMBED_DAEMON_IDLE_SEC seconds
of inactivity (default 900 = 15 min).

Run: python -m memory.embed_daemon [--idle-sec N]

Protocol (length-prefixed JSON, see embed_transport):

  Request:  {"op": "embed", "texts": [...], "id": "<reqid>"}
  Response: {"id": "<reqid>", "vectors": [[...], ...], "dim": 768}

  Request:  {"op": "ping"}
  Response: {"ok": true, "model": "...", "loaded": true, "uptime_sec": N}

  Request:  {"op": "shutdown", "token": "<pid>"}
  Response: {"ok": true}  (then exits)

Errors:     {"id": "<reqid>", "error": "...", "code": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from memory.embed_transport import (
    default_socket_path,
    listen,
    recv_msg,
    runtime_dir,
    send_msg,
)


DEFAULT_IDLE_SEC = int(os.environ.get("HERMES_EMBED_DAEMON_IDLE_SEC", "900"))
MAX_TEXTS_PER_REQUEST = 256
MAX_TEXT_BYTES = 32 * 1024


class EmbedDaemon:
    def __init__(self, socket_path: pathlib.Path, idle_sec: int):
        self.socket_path = socket_path
        self.idle_sec = idle_sec
        self.pid_path = runtime_dir() / "embed_daemon.pid"
        self.start_ts = time.monotonic()
        self.last_activity = time.monotonic()
        self.shutdown_flag = threading.Event()
        self._model = None
        self._model_lock = threading.Lock()
        self._stats = {"requests": 0, "errors": 0, "texts_embedded": 0}

    # ----- model lifecycle -----

    def _load_model(self):
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            # Use the shared EmbeddingService singleton path so the daemon
            # benefits from the same revision pinning + caching.
            from memory.embeddings import EmbeddingService
            svc = EmbeddingService()
            svc._load()
            self._model = svc._model

    # ----- PID file -----

    def _write_pid(self):
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        try:
            if sys.platform != "win32":
                os.chmod(self.pid_path, 0o600)
        except OSError:
            pass

    def _remove_pid(self):
        try:
            self.pid_path.unlink()
        except OSError:
            pass

    # ----- request handlers -----

    def _handle_embed(self, req: dict) -> dict:
        texts = req.get("texts", [])
        if not isinstance(texts, list) or not texts:
            return {"id": req.get("id"), "error": "missing or empty 'texts'",
                    "code": "BAD_REQUEST"}
        if len(texts) > MAX_TEXTS_PER_REQUEST:
            return {"id": req.get("id"),
                    "error": f"too many texts ({len(texts)} > {MAX_TEXTS_PER_REQUEST})",
                    "code": "BAD_REQUEST"}
        for t in texts:
            if not isinstance(t, str):
                return {"id": req.get("id"), "error": "non-string text",
                        "code": "BAD_REQUEST"}
            if len(t.encode("utf-8")) > MAX_TEXT_BYTES:
                return {"id": req.get("id"),
                        "error": f"text exceeds {MAX_TEXT_BYTES} bytes",
                        "code": "BAD_REQUEST"}

        self._load_model()
        try:
            with self._model_lock:
                vectors = self._model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()
        except Exception as e:
            return {"id": req.get("id"), "error": str(e)[:300],
                    "code": "ENCODE_FAILED"}

        self._stats["texts_embedded"] += len(texts)
        return {
            "id": req.get("id"),
            "vectors": vectors,
            "dim": len(vectors[0]) if vectors else 0,
        }

    def _handle_ping(self, req: dict) -> dict:
        from memory.config import EMBEDDING_MODEL
        return {
            "ok": True,
            "model": EMBEDDING_MODEL,
            "loaded": self._model is not None,
            "uptime_sec": time.monotonic() - self.start_ts,
            "idle_sec": time.monotonic() - self.last_activity,
            "stats": dict(self._stats),
            "pid": os.getpid(),
        }

    def _handle_shutdown(self, req: dict) -> dict:
        token = req.get("token", "")
        if token != str(os.getpid()):
            return {"error": "bad token", "code": "DENIED"}
        self.shutdown_flag.set()
        return {"ok": True}

    def _handle_request(self, req: dict) -> dict:
        op = req.get("op")
        self.last_activity = time.monotonic()
        self._stats["requests"] += 1
        if op == "embed":
            return self._handle_embed(req)
        if op == "ping":
            return self._handle_ping(req)
        if op == "shutdown":
            return self._handle_shutdown(req)
        self._stats["errors"] += 1
        return {"id": req.get("id"), "error": f"unknown op: {op!r}",
                "code": "BAD_REQUEST"}

    # ----- connection handling -----

    def _serve_connection(self, conn: socket.socket):
        try:
            while not self.shutdown_flag.is_set():
                try:
                    req = recv_msg(conn)
                except ValueError as e:
                    send_msg(conn, {"error": str(e), "code": "BAD_FRAME"})
                    return
                if req is None:
                    return  # client closed
                response = self._handle_request(req)
                try:
                    send_msg(conn, response)
                except OSError:
                    return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # ----- main loop -----

    def _idle_monitor(self):
        """Background thread: exit cleanly if no activity for idle_sec."""
        while not self.shutdown_flag.is_set():
            time.sleep(min(30, self.idle_sec))
            idle = time.monotonic() - self.last_activity
            if idle >= self.idle_sec:
                self._log(f"idle {idle:.0f}s >= {self.idle_sec}s, shutting down")
                self.shutdown_flag.set()
                # Wake the accept() loop by connecting once
                try:
                    from memory.embed_transport import connect
                    s = connect(self.socket_path, timeout=1.0)
                    s.close()
                except Exception:
                    pass
                break

    def _log(self, msg: str):
        log_path = runtime_dir() / "embed_daemon.log"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now(timezone.utc).isoformat()}] {msg}\n")
        except OSError:
            pass

    def run(self) -> int:
        self._write_pid()

        # Pre-warm the model so first client request is fast
        self._log(f"daemon starting (pid {os.getpid()}, idle_sec={self.idle_sec})")
        try:
            self._load_model()
            self._log("model warm")
        except Exception as e:
            self._log(f"model load failed: {e}")
            self._remove_pid()
            return 1

        # Open listener
        try:
            server = listen(self.socket_path)
        except OSError as e:
            self._log(f"listen failed: {e}")
            self._remove_pid()
            return 1

        # Idle monitor thread
        idle_thread = threading.Thread(target=self._idle_monitor, daemon=True)
        idle_thread.start()

        # Signal handlers
        def _term(signum, frame):
            self._log(f"received signal {signum}")
            self.shutdown_flag.set()
            try:
                server.close()
            except OSError:
                pass
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _term)
            except (OSError, ValueError):
                pass

        self._log(f"accepting on {self.socket_path}")
        server.settimeout(5.0)
        try:
            while not self.shutdown_flag.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                t = threading.Thread(target=self._serve_connection, args=(conn,),
                                       daemon=True)
                t.start()
        finally:
            try:
                server.close()
            except OSError:
                pass
            self._remove_pid()
            self._log("daemon stopped")

        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Hermes embedding daemon")
    ap.add_argument("--socket", help="Override socket / port-file path")
    ap.add_argument("--idle-sec", type=int, default=DEFAULT_IDLE_SEC)
    args = ap.parse_args()

    socket_path = pathlib.Path(args.socket) if args.socket else default_socket_path()
    daemon = EmbedDaemon(socket_path=socket_path, idle_sec=args.idle_sec)
    return daemon.run()


if __name__ == "__main__":
    sys.exit(main())
