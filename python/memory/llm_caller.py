"""Model-agnostic LLM caller for memory operations.

Supports multiple providers through AgentConfig:
  - ollama: Local Ollama instance (default, recommended)
  - openai: OpenAI API via OPENAI_API_KEY
  - anthropic-sdk: Anthropic API via ANTHROPIC_API_KEY
  - claude-cli: DISABLED — causes recursive MCP deadlock
  - custom: User-provided callable

The provider and model are configured per-agent in AgentConfig,
not hardcoded. Swap models by changing config, not code.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date
from typing import Callable

log = logging.getLogger(__name__)

LLMCaller = Callable[[str], str]
StructuredLLMCaller = Callable[[str], "dict | None"]


_caller_cache: dict[tuple, LLMCaller] = {}
_caller_cache_failures: dict[tuple, float] = {}
_RETRY_AFTER_SEC = 60.0


def get_llm_caller(
    provider: str = "claude-cli",
    model: str = "haiku",
    timeout: int = 120,
) -> LLMCaller | None:
    """Return a callable(prompt: str) -> str for the given provider.

    Args:
        provider: One of claude-cli, claude-cli-standalone, openai,
                  anthropic-sdk, ollama, custom.
        model: Model name/ID (interpretation depends on provider).
        timeout: Max seconds per call.

    Returns:
        Callable or None if the provider is unavailable.
    """
    key = (provider, model, timeout)

    if key in _caller_cache:
        return _caller_cache[key]

    last_fail = _caller_cache_failures.get(key, 0)
    if time.monotonic() - last_fail < _RETRY_AFTER_SEC:
        return None

    builders = {
        "claude-cli": _build_claude_cli,
        "claude-cli-standalone": _build_claude_cli_standalone,
        "openai": _build_openai,
        "anthropic-sdk": _build_anthropic,
        "ollama": _build_ollama,
    }

    builder = builders.get(provider)
    if builder is None:
        log.warning(f"Unknown LLM provider: {provider}")
        return None

    result = builder(model, timeout)
    if result is None:
        _caller_cache_failures[key] = time.monotonic()
        log.warning(f"LLM provider {provider}/{model} unavailable, will retry after {_RETRY_AFTER_SEC}s")
        return None

    _caller_cache[key] = result
    return result


def get_llm_caller_for_agent(agent_name: str) -> LLMCaller | None:
    """Convenience: get caller using the agent's registered config."""
    from memory.agent_config import get_agent_config
    config = get_agent_config(agent_name)
    return get_llm_caller(config.llm_provider, config.llm_model, config.llm_timeout)


# --- Provider implementations ---

def _build_claude_cli(model: str, timeout: int) -> LLMCaller | None:
    # UNSAFE: spawning `claude -p` from inside an MCP tool handler causes recursive
    # deadlock — child inherits .mcp.json, spawns another memory_server, both block.
    # Use ollama, openai, or anthropic-sdk instead.
    log.error("claude-cli provider is disabled — causes recursive MCP deadlock when "
              "called from within an MCP tool handler. Use ollama provider instead.")
    return None


def _build_claude_cli_standalone(model: str, timeout: int) -> LLMCaller | None:
    """Claude CLI for standalone processes (NOT MCP tool handlers).

    Safe when called from background workers, cron jobs, or any process
    that doesn't inherit .mcp.json context. Uses the user's Claude
    subscription quota (not API keys).

    Invokes: claude --print --model {model} with prompt on stdin.
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        log.warning("claude CLI not found in PATH — claude-cli-standalone unavailable")
        return None

    def call(prompt: str) -> str:
        cmd = [claude_bin, "--print", "--model", model]
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if platform.system() == "Windows" else 0),
                env={**os.environ, "CLAUDE_CODE_ENTRYPOINT": "standalone-worker",
                     "LETTA_MODE": "off", "HERMES_MODE": "off"},
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()[:200] if result.stderr else ""
                log.warning("claude CLI returned %d: %s", result.returncode, stderr)
                return ""
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            log.warning("claude CLI timed out after %ds", timeout)
            return ""
        except OSError as e:
            log.error("claude CLI exec failed: %s", e)
            return ""

    # Quick validation: check that claude --version works
    try:
        check = subprocess.run(
            [claude_bin, "--version"],
            capture_output=True, text=True, encoding="utf-8", timeout=10,
            creationflags=(subprocess.CREATE_NO_WINDOW
                           if platform.system() == "Windows" else 0),
        )
        if check.returncode != 0:
            log.warning("claude --version failed (rc=%d) — provider unavailable", check.returncode)
            return None
        log.info("claude-cli-standalone ready: %s", check.stdout.strip()[:60])
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("claude CLI validation failed: %s", e)
        return None

    return call


def _build_openai(model: str, timeout: int) -> LLMCaller | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.warning("OPENAI_API_KEY not set — OpenAI provider unavailable")
        return None

    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=timeout)

    def call(prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"OpenAI error: {e}")
            return ""

    return call


def _build_anthropic(model: str, timeout: int) -> LLMCaller | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set — Anthropic SDK unavailable")
        return None

    import anthropic
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def call(prompt: str) -> str:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            log.error(f"Anthropic SDK error: {e}")
            return ""

    return call


def _ensure_ollama_running() -> bool:
    """Check if Ollama is running; if not, try to start it.

    Returns True if Ollama is reachable, False otherwise.
    Uses a file lock to prevent multiple callers from spawning Ollama simultaneously.
    """
    import requests as _req
    from filelock import FileLock, Timeout

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    # Probe existing instance
    try:
        resp = _req.get(f"{ollama_host}/api/tags", timeout=2)
        if resp.ok:
            log.info("Ollama already running")
            return True
    except Exception:
        pass

    # Try to start Ollama — locked to prevent duplicate spawns
    exe = shutil.which("ollama")
    if not exe:
        log.warning("Ollama executable not found")
        return False

    lock_path = pathlib.Path.home() / ".hermes" / "runtime" / "ollama_spawn.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(lock_path), timeout=5):
            # Re-check after acquiring lock
            try:
                resp = _req.get(f"{ollama_host}/api/tags", timeout=2)
                if resp.ok:
                    log.info("Ollama already running (after lock)")
                    return True
            except Exception:
                pass

            log.info(f"Starting Ollama via {exe}")
            try:
                creation_flags = 0
                if platform.system() == "Windows":
                    creation_flags = subprocess.CREATE_NO_WINDOW
                subprocess.Popen(
                    [exe, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creation_flags,
                )
            except Exception as e:
                log.warning(f"Failed to start Ollama: {e}")
                return False
    except Timeout:
        log.info("Another caller is spawning Ollama, waiting...")
        # Just wait for the other spawner to finish

    # Wait for it to come up
    for _ in range(10):
        time.sleep(0.5)
        try:
            resp = _req.get(f"{ollama_host}/api/tags", timeout=2)
            if resp.ok:
                log.info("Ollama started successfully")
                return True
        except Exception:
            continue

    log.warning("Ollama did not start in time")
    return False


def _build_ollama(model: str, timeout: int) -> LLMCaller | None:
    if not _ensure_ollama_running():
        log.warning("Ollama unavailable — enrichment will be skipped")
        return None

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def call(prompt: str) -> str:
        try:
            import requests
            response = requests.post(
                f"{ollama_host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            log.error(f"Ollama error: {e}")
            return ""

    return call


# --- Structured-output caller (Claude CLI --json-schema) ---

_COST_WARN_THRESHOLD = float(os.environ.get("HERMES_LLM_COST_WARN_USD", "5.0"))


def _cost_ledger_path() -> pathlib.Path:
    return pathlib.Path.home() / ".hermes" / "runtime" / "cost_ledger.json"


def _record_cost(usd: float) -> None:
    """Append a cost record to the persistent daily ledger. Cross-process safe."""
    if not isinstance(usd, (int, float)) or usd <= 0:
        return
    path = _cost_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    try:
        from filelock import FileLock, Timeout
    except ImportError:
        log.debug("filelock not available — skipping cost ledger update")
        return
    try:
        with FileLock(str(path) + ".lock", timeout=2):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                data = {}
            if data.get("date") != today:
                data = {"date": today, "total_usd": 0.0}
            data["total_usd"] = round(data.get("total_usd", 0.0) + float(usd), 6)
            path.write_text(json.dumps(data), encoding="utf-8")
            if data["total_usd"] > _COST_WARN_THRESHOLD:
                log.warning(
                    "Hermes LLM cost today: $%.2f (threshold $%.2f)",
                    data["total_usd"], _COST_WARN_THRESHOLD,
                )
    except Timeout:
        log.debug("cost ledger locked, skipping update")


def _sacrificial_config_dir() -> pathlib.Path:
    """Return a stable config directory containing ONLY credentials.

    Forces claude --print to read config from this dir rather than from the
    user's real ~/.claude/. We copy `.credentials.json` (needed for OAuth/
    subscription auth) but deliberately DO NOT copy `settings.json` — that's
    where `enabledPlugins` lives, and inheriting it would trigger recursive
    plugin loading (POSTMORTEM_2026-05-11).

    Credentials are refreshed when the source file is newer than the copy,
    so re-auth in the main session propagates without manual intervention.
    """
    p = pathlib.Path.home() / ".hermes" / "runtime" / "sacrificial_claude_config"
    p.mkdir(parents=True, exist_ok=True)
    src = pathlib.Path.home() / ".claude" / ".credentials.json"
    if src.exists():
        dst = p / ".credentials.json"
        try:
            src_mtime = src.stat().st_mtime
            if not dst.exists() or dst.stat().st_mtime < src_mtime:
                shutil.copy2(src, dst)
        except OSError as e:
            log.warning("Failed to sync sacrificial credentials: %s", e)
    return p


def get_structured_llm_caller(
    provider: str,
    model: str,
    schema: dict,
    system_prompt: str,
    timeout: int = 90,
) -> StructuredLLMCaller | None:
    """Return callable(user_msg: str) -> dict | None for structured output.

    Currently only `claude-cli-standalone` is supported. The returned dict
    matches `schema`, or None on failure.
    """
    if provider != "claude-cli-standalone":
        log.warning(
            "Structured output only supported for 'claude-cli-standalone' provider, got %s",
            provider,
        )
        return None
    return _build_claude_cli_structured(model, schema, system_prompt, timeout)


def _build_claude_cli_structured(
    model: str,
    schema: dict,
    system_prompt: str,
    timeout: int,
) -> StructuredLLMCaller | None:
    claude_bin = shutil.which("claude")
    if not claude_bin:
        log.warning("claude CLI not found in PATH — structured output unavailable")
        return None

    # Probe: verify --json-schema flag exists in this CLI version.
    try:
        help_out = subprocess.run(
            [claude_bin, "--print", "--help"],
            capture_output=True, text=True, timeout=10,
            encoding="utf-8",
            creationflags=(subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0),
        )
        if "--json-schema" not in (help_out.stdout or ""):
            log.error(
                "claude CLI lacks --json-schema flag — structured output unavailable. "
                "Update Claude Code to a version that supports structured output."
            )
            return None
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("claude --help probe failed: %s", e)
        return None

    schema_json = json.dumps(schema)
    config_dir = str(_sacrificial_config_dir())

    def call(user_msg: str) -> "dict | None":
        # cwd MUST be outside any directory containing CLAUDE.md or .claude/.
        # On Windows the user temp is inside the user profile, so claude would
        # still walk up to ~/.claude/. We override via CLAUDE_CONFIG_DIR.
        #
        # System prompt + schema are written to files because long inline
        # --system-prompt arguments silently fail on Windows once the command
        # line exceeds ~32k chars (CreateProcess limit). When that happens
        # claude returns markdown-wrapped JSON instead of the structured envelope.
        tmpcwd = tempfile.mkdtemp(prefix="hermes-loop-")
        sys_prompt_path = os.path.join(tmpcwd, "system_prompt.txt")
        try:
            with open(sys_prompt_path, "w", encoding="utf-8") as f:
                f.write(system_prompt)
            env = {k: v for k, v in os.environ.items()}
            env.pop("PYTHONPATH", None)
            env["CLAUDE_CODE_ENTRYPOINT"] = "standalone-worker"
            env["CLAUDE_CONFIG_DIR"] = config_dir
            env["LETTA_MODE"] = "off"
            env["HERMES_MODE"] = "off"
            try:
                result = subprocess.run(
                    [claude_bin, "--print", "--model", model,
                     "--system-prompt-file", sys_prompt_path,
                     "--json-schema", schema_json,
                     "--output-format", "json"],
                    input=user_msg,
                    capture_output=True, text=True, timeout=timeout,
                    encoding="utf-8",
                    cwd=tmpcwd,
                    env=env,
                    creationflags=(subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0),
                )
            except subprocess.TimeoutExpired:
                log.warning("claude structured call timed out after %ds", timeout)
                return None
            except OSError as e:
                log.error("claude exec failed: %s", e)
                return None
        finally:
            shutil.rmtree(tmpcwd, ignore_errors=True)

        if result.returncode != 0:
            stderr_msg = (result.stderr or "")[:300]
            log.warning("claude returned %d: %s", result.returncode, stderr_msg)
            return None

        try:
            envelope = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            log.warning("invalid envelope JSON: %s, raw: %s", e, (result.stdout or "")[:200])
            return None

        if envelope.get("is_error"):
            log.warning("claude reported error: %s", envelope.get("api_error_status"))
            return None

        cost = envelope.get("total_cost_usd")
        if isinstance(cost, (int, float)):
            _record_cost(cost)

        structured = envelope.get("structured_output")
        if not isinstance(structured, dict):
            log.warning(
                "missing or non-dict structured_output in envelope; result preview: %s",
                str(envelope.get("result"))[:200],
            )
            return None
        return structured

    return call
