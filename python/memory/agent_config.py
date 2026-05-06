"""Per-agent configuration — the single source of truth for each node.

Every agent (Hermes, Atlas, Lit, etc.) gets an AgentConfig that controls:
  - Which LLM model and provider to use
  - Which namespaces to read from
  - Triple-score weights and retrieval parameters
  - Whether enrichment/evolution are enabled
  - Reflection thresholds

Config is model-agnostic: any LLM provider can be plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memory.config import (
    ALPHA_RELEVANCE,
    BETA_RECENCY,
    GAMMA_IMPORTANCE,
    RECENCY_DECAY_RATE,
)


def _load_tuned_weights() -> tuple[float, float, float] | None:
    """Try to load self-tuned weights. Returns None if unavailable."""
    try:
        from memory.self_tune import get_current_weights
        return get_current_weights()
    except (ImportError, Exception):
        return None


@dataclass
class AgentConfig:
    """Configuration for a single agent node."""

    agent_name: str

    # LLM settings — provider-agnostic
    llm_provider: str = "ollama"  # claude-cli | openai | anthropic-sdk | ollama | custom
    llm_model: str = "qwen2.5:1.5b"
    llm_timeout: int = 30

    # Namespace access control
    namespaces: list[str] = field(default_factory=lambda: ["shared"])
    write_namespace: str = ""  # defaults to agent_name if empty

    # Feature toggles
    enrich_on_insert: bool = True
    evolve_on_insert: bool = False
    reflection_enabled: bool = True
    reflection_threshold: float = 15.0

    # Triple-score weights
    alpha_relevance: float = ALPHA_RELEVANCE
    beta_recency: float = BETA_RECENCY
    gamma_importance: float = GAMMA_IMPORTANCE
    recency_decay_rate: float = RECENCY_DECAY_RATE

    # Bootstrap settings
    l1_pinned_max: int = 5
    l1_vector_budget: int = 10
    l1_total_budget: int = 15
    bootstrap_header: str = ""  # defaults to "# {agent_name} Memory Context"
    domain_modules: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.write_namespace:
            self.write_namespace = self.agent_name
        if self.agent_name not in self.namespaces:
            self.namespaces = [self.agent_name] + self.namespaces
        if not self.bootstrap_header:
            self.bootstrap_header = f"# {self.agent_name.title()} Memory Context"

        # Load self-tuned weights if available (overrides defaults)
        tuned = _load_tuned_weights()
        if tuned:
            self.alpha_relevance, self.beta_recency, self.gamma_importance = tuned


# --- Pre-built configs for known agents ---

HERMES = AgentConfig(
    agent_name="hermes",
    llm_provider="ollama",
    llm_model="qwen2.5:1.5b",
    namespaces=["hermes", "shared", "auto_pricer", "migration"],
    enrich_on_insert=True,
    evolve_on_insert=False,
    domain_modules=["ml.model_registry"],
)

ATLAS = AgentConfig(
    agent_name="atlas",
    llm_provider="ollama",
    llm_model="qwen2.5:1.5b",
    namespaces=["atlas", "shared"],
    enrich_on_insert=True,
    evolve_on_insert=False,
    domain_modules=[],
)

LIT = AgentConfig(
    agent_name="lit",
    llm_provider="ollama",
    llm_model="qwen2.5:1.5b",
    namespaces=["lit", "shared"],
    enrich_on_insert=False,
    evolve_on_insert=False,
    reflection_enabled=False,
)

AGENT_REGISTRY: dict[str, AgentConfig] = {
    "hermes": HERMES,
    "atlas": ATLAS,
    "lit": LIT,
}


def get_agent_config(agent_name: str) -> AgentConfig:
    """Get config for a known agent, or create a default one."""
    if agent_name in AGENT_REGISTRY:
        return AGENT_REGISTRY[agent_name]
    return AgentConfig(agent_name=agent_name)
