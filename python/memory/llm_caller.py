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
import logging
import os
import platform
import shutil
import subprocess
import time
from typing import Callable

log = logging.getLogger(__name__)

LLMCaller = Callable[[str], str]


@functools.lru_cache(maxsize=4)
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

    return builder(model, timeout)


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
                timeout=timeout,
                # NOTE: DETACHED_PROCESS removed 2026-05-12 — it broke piped
                # stdin/stdout to claude --print on Windows (capture_output
                # pipes need a console context). Was added in 5da9367 based
                # on a phantom recursive-chain hypothesis that turned out to
                # be a diagnostic self-match (see POSTMORTEM_2026-05-11.md).
                # CREATE_NO_WINDOW alone suppresses console flash and is safe.
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
            capture_output=True, text=True, timeout=10,
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
    """
    import requests as _req

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    # Probe existing instance
    try:
        resp = _req.get(f"{ollama_host}/api/tags", timeout=2)
        if resp.ok:
            log.info("Ollama already running")
            return True
    except Exception:
        pass

    # Try to start Ollama
    exe = shutil.which("ollama")
    if not exe:
        log.warning("Ollama executable not found")
        return False

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
