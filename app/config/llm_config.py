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

from typing import Any
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMFactoryConfig(BaseSettings):
    """Configuration for LLM models, retries, timeouts, and embeddings in the factory."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    primary_model: str = Field(
        default="openai/gpt-oss-20b", validation_alias="PRIMARY_LLM_MODEL"
    )
    fallback_1_model: str = Field(
        default="openai/gpt-oss-120b", validation_alias="FALLBACK_1_LLM_MODEL"
    )
    fallback_2_model: str = Field(
        default="gemini-3.1-flash-lite", validation_alias="FALLBACK_2_LLM_MODEL"
    )
    max_retries: int = Field(
        default=3, validation_alias="LLM_MAX_RETRIES"
    )
    timeout: float = Field(
        default=15.0, validation_alias="LLM_TIMEOUT"
    )
    primary_embedding_model: str = Field(
        default="openai/text-embedding-3-small", validation_alias="OPENROUTER_EMBEDDING_MODEL"
    )
    fallback_embedding_model: str = Field(
        default="models/gemini-embedding-2", validation_alias="EMBEDDING_MODEL"
    )


LLM_FACTORY_CONFIG = LLMFactoryConfig()


class LLMConfig(BaseModel):
    """Immutable LLM configuration for a single agent.

    Attributes:
        temperature:    Sampling temperature passed to the LLM.
        default_cache:  Cache behaviour when no runtime override applies.
                        None  → use the global LangChain cache (if configured).
                        False → always disable the cache for this agent.
                        True  → force-enable the cache for this agent.
    """
    model_config = {"frozen": True}

    temperature: float
    default_cache: bool | None


class AllAgentLLMSettings(BaseSettings):
    """Configuration class to parse all per-agent LLM overrides from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    supervisor_llm_temperature: float = Field(
        default=0.0, validation_alias="SUPERVISOR_LLM_TEMPERATURE"
    )
    supervisor_llm_cache: bool | None = Field(
        default=None, validation_alias="SUPERVISOR_LLM_CACHE"
    )

    faq_llm_temperature: float = Field(
        default=0.4, validation_alias="FAQ_LLM_TEMPERATURE"
    )
    faq_llm_cache: bool | None = Field(
        default=None, validation_alias="FAQ_LLM_CACHE"
    )

    order_llm_temperature: float = Field(
        default=0.1, validation_alias="ORDER_LLM_TEMPERATURE"
    )
    order_llm_cache: bool | None = Field(
        default=False, validation_alias="ORDER_LLM_CACHE"
    )

    summarizer_llm_temperature: float = Field(
        default=0.0, validation_alias="SUMMARIZER_LLM_TEMPERATURE"
    )
    summarizer_llm_cache: bool | None = Field(
        default=None, validation_alias="SUMMARIZER_LLM_CACHE"
    )

    synthesizer_llm_temperature: float = Field(
        default=0.3, validation_alias="SYNTHESIZER_LLM_TEMPERATURE"
    )
    synthesizer_llm_cache: bool | None = Field(
        default=None, validation_alias="SYNTHESIZER_LLM_CACHE"
    )

    @field_validator(
        "supervisor_llm_cache",
        "faq_llm_cache",
        "order_llm_cache",
        "summarizer_llm_cache",
        "synthesizer_llm_cache",
        mode="before"
    )
    @classmethod
    def coerce_none_string(cls, v: Any) -> Any:
        """Coerce string values representing null/none into Python None."""
        if isinstance(v, str) and v.lower() in ("none", "null", ""):
            return None
        return v


_agent_settings = AllAgentLLMSettings()


# ── Per-agent configs ────────────────────────────────────────────────────────

SUPERVISOR_LLM_CONFIG = LLMConfig(
    temperature=_agent_settings.supervisor_llm_temperature,
    default_cache=_agent_settings.supervisor_llm_cache,
)

FAQ_LLM_CONFIG = LLMConfig(
    temperature=_agent_settings.faq_llm_temperature,
    default_cache=_agent_settings.faq_llm_cache,
)

ORDER_LLM_CONFIG = LLMConfig(
    temperature=_agent_settings.order_llm_temperature,
    default_cache=_agent_settings.order_llm_cache,
)

SUMMARIZER_LLM_CONFIG = LLMConfig(
    temperature=_agent_settings.summarizer_llm_temperature,
    default_cache=_agent_settings.summarizer_llm_cache,
)

SYNTHESIZER_LLM_CONFIG = LLMConfig(
    temperature=_agent_settings.synthesizer_llm_temperature,
    default_cache=_agent_settings.synthesizer_llm_cache,
)


class AgentConfig(BaseSettings):
    """General configuration for agent loops."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_iterations: int = Field(default=6, validation_alias="MAX_ITERATIONS")


AGENT_CONFIG = AgentConfig()
