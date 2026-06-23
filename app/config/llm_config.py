"""
Centralized LLM configuration for all agents.

All per-agent temperature and cache defaults live here so they can be
tuned from a single file without touching agent logic.

Usage in agents:
    from app.config.llm_config import SUPERVISOR_LLM_CONFIG
    llm = get_llm(
        temperature=SUPERVISOR_LLM_CONFIG.temperature,
        cache=False if has_image_context else SUPERVISOR_LLM_CONFIG.default_cache,
    )

For agents with no dynamic image-context override (e.g. order, summarizer):
    llm = get_llm(
        temperature=ORDER_LLM_CONFIG.temperature,
        cache=ORDER_LLM_CONFIG.default_cache,
    )
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMFactoryConfig:
    """Configuration for LLM models, retries, timeouts, and embeddings in the factory."""

    primary_model: str = os.getenv("PRIMARY_LLM_MODEL", "openai/gpt-oss-20b")
    fallback_1_model: str = os.getenv("FALLBACK_1_LLM_MODEL", "openai/gpt-oss-120b")
    fallback_2_model: str = os.getenv("FALLBACK_2_LLM_MODEL", "gemini-3.1-flash-lite")
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    timeout: float = float(os.getenv("LLM_TIMEOUT", "15.0"))
    primary_embedding_model: str = os.getenv(
        "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
    )
    fallback_embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "models/gemini-embedding-2"
    )


LLM_FACTORY_CONFIG = LLMFactoryConfig()


@dataclass(frozen=True)
class LLMConfig:
    """Immutable LLM configuration for a single agent.

    Attributes:
        temperature:    Sampling temperature passed to the LLM.
        default_cache:  Cache behaviour when no runtime override applies.
                        None  → use the global LangChain cache (if configured).
                        False → always disable the cache for this agent.
                        True  → force-enable the cache for this agent.
    """

    temperature: float
    default_cache: bool | None


# ── Per-agent configs ────────────────────────────────────────────────────────

# Routing only — deterministic, cache enabled when no image is present.
SUPERVISOR_LLM_CONFIG = LLMConfig(temperature=0.0, default_cache=None)

# FAQ answers — slightly creative, cache enabled for repeated policy questions.
FAQ_LLM_CONFIG = LLMConfig(temperature=0.4, default_cache=None)

# Order lookups — always fetches live DB data, never cache.
ORDER_LLM_CONFIG = LLMConfig(temperature=0.1, default_cache=False)

# Summarization — fully deterministic, no image context, global cache applies.
SUMMARIZER_LLM_CONFIG = LLMConfig(temperature=0.0, default_cache=None)

# Synthesis — slightly creative to produce natural merged replies.
SYNTHESIZER_LLM_CONFIG = LLMConfig(temperature=0.3, default_cache=None)
