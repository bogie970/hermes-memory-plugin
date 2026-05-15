"""Cross-platform IPC transport for the embedding daemon.

Length-prefixed JSON over stream socket:
    [4 bytes big-endian length] [JSON payload]

POSIX:   Unix domain socket at ~/.hermes/runtime/embed.sock (mode 0700)
Windows: TCP loopback at 127.0.0.1:<port>; port written to ~/.hermes/runtime/embed.port

We use TCP loopback on Windows rather than named pipes because:
  - named pipes require pywin32 or ctypes scaffolding for stream semantics
  - TCP loopback is built-in, no extra deps
  - localhost-only binding + single-user box means same security
    posture as a Unix socket with 0700 perms
  - the port file is mode 0600 (Windows ACL) so other users can't read it

Module-level helpers:
    listen(socket_path)   -> server socket
    connect(socket_path)  -> client socket
    send_msg(sock, dict)  -> length-prefixed JSON write
    recv_msg(sock)        -> length-prefixed JSON read (returns dict or None on EOF)
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
import struct
import sys

IS_WINDOWS = sys.platform == "win32"
MAX_MESSAGE_BYTES = 8 * 1024 * 1024  # 8 MB hard cap
EMBED_DAEMON_PORT = int(os.environ.get("HERMES_EMBED_DAEMON_PORT", "19384"))


def runtime_dir() -> pathlib.Path:
    """Resolve ~/.hermes/runtime/, create with restrictive perms."""
    data_dir = pathlib.Path(
        os.environ.get("HERMES_DATA_DIR")
        or pathlib.Path.home() / ".hermes"
    )
    rt = data_dir / "runtime"
    rt.mkdir(parents=True, exist_ok=True)
    if not IS_WINDOWS:
        try:
            os.chmod(rt, 0o700)
        except OSError:
            pass
    return rt


def default_socket_path() -> pathlib.Path:
    """Where the daemon listens. POSIX uses a socket file; Windows uses a port file."""
    rt = runtime_dir()
    if IS_WINDOWS:
        return rt / "embed.port"
    return rt / "embed.sock"


# ---- Server-side ----

def listen(socket_path: pathlib.Path) -> socket.socket:
    """Open a listening server socket. Returns the socket; caller accepts.

    POSIX:   binds to socket_path (unlinking first if stale)
    Windows: binds to 127.0.0.1:0 (random port); writes the port number
             into socket_path as a single-line text file (mode 0600)
    """
    if IS_WINDOWS:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # SO_EXCLUSIVEADDRUSE prevents a second daemon from binding the same port
        s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)  # type: ignore[attr-defined]
        s.bind(("127.0.0.1", EMBED_DAEMON_PORT))
        s.listen(8)
        port = s.getsockname()[1]
        socket_path.write_text(str(port), encoding="utf-8")
        return s
    else:
        if socket_path.exists():
            try:
                socket_path.unlink()
            except OSError:
                pass
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(socket_path))
        try:
            os.chmod(socket_path, 0o700)
        except OSError:
            pass
        s.listen(8)
        return s


# ---- Client-side ----

def connect(socket_path: pathlib.Path, timeout: float = 2.0) -> socket.socket:
    """Connect to the daemon. Raises ConnectionError if unreachable.

    Caller is responsible for closing the socket.
    """
    if IS_WINDOWS:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(("127.0.0.1", EMBED_DAEMON_PORT))
        s.settimeout(None)
        return s
    else:
        if not socket_path.exists():
            raise ConnectionError(f"socket missing: {socket_path}")
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(str(socket_path))
        s.settimeout(None)
        return s


# ---- Framing ----

def send_msg(sock: socket.socket, payload: dict) -> None:
    """Encode payload as JSON, send with 4-byte length prefix."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message too large: {len(data)} > {MAX_MESSAGE_BYTES}")
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes from sock, or None on EOF/error."""
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            raise
        except (ConnectionResetError, OSError):
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket, timeout: float | None = 60.0) -> dict | None:
    """Read one length-prefixed JSON message. Returns None on EOF.

    Raises ValueError on framing/size errors.
    Raises socket.timeout if no data arrives within `timeout` seconds.
    Pass timeout=None to wait indefinitely (not recommended).
    """
    old_timeout = sock.gettimeout()
    if timeout is not None:
        sock.settimeout(timeout)
    try:
        header = _recv_exact(sock, 4)
        if header is None:
            return None
        (length,) = struct.unpack(">I", header)
        if length == 0:
            return {}
        if length > MAX_MESSAGE_BYTES:
            raise ValueError(f"oversized message: {length}")
        body = _recv_exact(sock, length)
        if body is None:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"bad json: {e}")
    finally:
        sock.settimeout(old_timeout)
