"""Tests for the embedding daemon, client, and transport layer.

Strategy: pure-Python tests against the transport + protocol layers
WITHOUT loading the real 500MB model. We test:
  - transport framing (length-prefixed JSON, oversize rejection)
  - client spawn / connect / fallback
  - PID-file management
  - server request dispatch with a mocked model

E2E tests with the real model are deferred to manual smoke testing.

Run: pytest aisys/memory/tests/test_embed_daemon.py -v
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Isolate the runtime dir so tests don't touch the real daemon."""
    monkeypatch.setenv("HERMES_DATA_DIR", str(tmp_path))
    return tmp_path


# ----- transport: framing -----

def test_send_recv_msg_roundtrip(isolated):
    """A small dict roundtrips intact through send_msg + recv_msg."""
    from memory.embed_transport import recv_msg, send_msg
    a, b = socket.socketpair()
    try:
        payload = {"op": "ping", "id": "abc123", "nested": {"k": [1, 2, 3]}}
        send_msg(a, payload)
        decoded = recv_msg(b)
        assert decoded == payload
    finally:
        a.close()
        b.close()


def test_recv_msg_returns_none_on_eof(isolated):
    """Reading from a closed peer returns None (clean EOF)."""
    from memory.embed_transport import recv_msg
    a, b = socket.socketpair()
    a.close()
    assert recv_msg(b) is None
    b.close()


def test_send_msg_rejects_oversize(isolated):
    """Payload over MAX_MESSAGE_BYTES raises ValueError before send."""
    from memory.embed_transport import MAX_MESSAGE_BYTES, send_msg
    a, b = socket.socketpair()
    try:
        huge = {"data": "x" * (MAX_MESSAGE_BYTES + 1)}
        with pytest.raises(ValueError):
            send_msg(a, huge)
    finally:
        a.close()
        b.close()


def test_recv_msg_rejects_oversize_header(isolated):
    """Length prefix above cap raises ValueError without allocating."""
    from memory.embed_transport import MAX_MESSAGE_BYTES, recv_msg
    a, b = socket.socketpair()
    try:
        bad_length = MAX_MESSAGE_BYTES + 1
        a.sendall(struct.pack(">I", bad_length))
        with pytest.raises(ValueError):
            recv_msg(b)
    finally:
        a.close()
        b.close()


# ----- transport: socket / port file path -----

def test_default_socket_path_uses_runtime_dir(isolated):
    from memory.embed_transport import default_socket_path, runtime_dir
    p = default_socket_path()
    assert p.parent == runtime_dir()


# ----- client: pid alive check -----

def test_pid_alive_for_current_process():
    from memory.embed_client import _pid_alive
    assert _pid_alive(os.getpid())


def test_pid_alive_false_for_zero():
    from memory.embed_client import _pid_alive
    assert not _pid_alive(0)


def test_pid_alive_false_for_unlikely_pid():
    from memory.embed_client import _pid_alive
    # PID 99999999 unlikely to exist
    assert not _pid_alive(99999999)


# ----- client: stale PID file cleanup -----

def test_daemon_alive_returns_false_no_pid_file(isolated):
    from memory.embed_client import DaemonClient
    client = DaemonClient()
    assert not client._daemon_alive()


def test_daemon_alive_cleans_stale_pid_file(isolated):
    """Stale PID file (pid not running) gets unlinked on detection."""
    from memory.embed_client import DaemonClient
    client = DaemonClient()
    # Write a fake PID for a long-dead process
    client.pid_path.write_text("99999999")
    assert not client._daemon_alive()
    # Stale file should be cleaned
    assert not client.pid_path.exists()


# ----- end-to-end: spin a daemon with a MOCKED model -----

def test_daemon_serves_embed_request_with_fake_model(isolated, monkeypatch):
    """End-to-end: client connects to a real daemon with a mocked model."""
    from memory.embed_daemon import EmbedDaemon
    from memory.embed_transport import default_socket_path

    # Build daemon
    socket_path = default_socket_path()
    daemon = EmbedDaemon(socket_path=socket_path, idle_sec=60)
    # Stub the model so we don't load 500MB
    class FakeModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            import numpy as np
            # Return deterministic vectors based on hash of text
            return np.array([
                [(hash(t) % 1000) / 1000.0] * 8 for t in texts
            ])
    daemon._model = FakeModel()

    # Run the daemon's accept loop in a background thread
    def serve():
        try:
            daemon._write_pid()
            from memory.embed_transport import listen
            server = listen(socket_path)
            server.settimeout(5.0)
            try:
                while not daemon.shutdown_flag.is_set():
                    try:
                        conn, _ = server.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    daemon._serve_connection(conn)
            finally:
                server.close()
        except Exception as e:
            print(f"daemon thread crashed: {e}")

    t = threading.Thread(target=serve, daemon=True)
    t.start()

    # Wait for PID file to appear (daemon ready)
    for _ in range(50):
        if daemon.pid_path.exists():
            break
        time.sleep(0.05)

    # Client embeds
    try:
        from memory.embed_client import DaemonClient
        client = DaemonClient(socket_path=socket_path)
        vectors = client.embed(["hello", "world"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 8

        # Ping
        status = client.ping()
        assert status is not None
        assert status["ok"]
    finally:
        daemon.shutdown_flag.set()
        # Give thread a moment to exit
        time.sleep(0.2)


def test_daemon_handles_bad_op_gracefully(isolated):
    """Unknown op returns an error response, not a crash."""
    from memory.embed_daemon import EmbedDaemon
    from memory.embed_transport import default_socket_path

    daemon = EmbedDaemon(socket_path=default_socket_path(), idle_sec=60)
    daemon._model = object()  # any non-None placeholder

    req = {"op": "fly_to_mars", "id": "x"}
    response = daemon._handle_request(req)
    assert "error" in response
    assert response["code"] == "BAD_REQUEST"


def test_daemon_rejects_too_many_texts(isolated):
    from memory.embed_daemon import EmbedDaemon, MAX_TEXTS_PER_REQUEST
    from memory.embed_transport import default_socket_path

    daemon = EmbedDaemon(socket_path=default_socket_path(), idle_sec=60)
    response = daemon._handle_embed({
        "id": "x",
        "texts": ["t"] * (MAX_TEXTS_PER_REQUEST + 1),
    })
    assert response["code"] == "BAD_REQUEST"


def test_daemon_shutdown_requires_token(isolated):
    from memory.embed_daemon import EmbedDaemon
    from memory.embed_transport import default_socket_path

    daemon = EmbedDaemon(socket_path=default_socket_path(), idle_sec=60)
    bad = daemon._handle_shutdown({"token": "wrong"})
    assert bad["code"] == "DENIED"
    assert not daemon.shutdown_flag.is_set()

    good = daemon._handle_shutdown({"token": str(os.getpid())})
    assert good.get("ok")
    assert daemon.shutdown_flag.is_set()


# ----- fallback: HERMES_EMBED_DAEMON=0 disables daemon path -----

def test_embedding_service_skips_daemon_when_disabled(isolated, monkeypatch):
    """HERMES_EMBED_DAEMON=0 -> EmbeddingService never tries daemon."""
    monkeypatch.setenv("HERMES_EMBED_DAEMON", "0")
    from memory.embeddings import EmbeddingService
    svc = EmbeddingService()
    assert svc._daemon_disabled
    # _try_daemon should short-circuit without spawn attempt
    result = svc._try_daemon(["test"])
    assert result is None
