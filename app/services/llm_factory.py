"""
Factory functions for creating LLM and Embedding model instances with fallback chains.

Usage
-----
LLM (with automatic fallback):
    from app.services.llm_factory import get_llm
    llm = get_llm(temperature=0.4)

Embeddings (with automatic fallback):
    from app.services.llm_factory import get_embeddings
    embeddings = get_embeddings(dimensions=512)

Fallback strategy
-----------------
LLMs:    primary (fast model) → fallback_1 (large model) → fallback_2 (Gemini)
Embeds:  primary (OpenRouter text-embedding-3-small) → fallback (Gemini Embedding)
"""

import logging
import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

from app.config import LLM_FACTORY_CONFIG

load_dotenv()

logger = logging.getLogger(__name__)


# ── Embedding fallback wrapper ────────────────────────────────────────────────


class FallbackEmbeddings(Embeddings):
    """
    Embedding model wrapper that transparently falls back to a secondary model.

    If the primary model raises any exception (e.g. rate limit, network error),
    the request is automatically retried against the fallback model. Both sync
    and async variants are supported.

    Args:
        primary: The preferred embedding model instance.
        fallback: The backup embedding model instance.
        primary_name: Human-readable name used in log messages.
        fallback_name: Human-readable name used in log messages.
    """

    def __init__(self, primary, fallback, primary_name: str, fallback_name: str):
        self.primary = primary
        self.fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name

    def embed_documents(self, texts):
        """
        Generate embeddings for a list of text documents (sync).

        Tries the primary model first, falls back on any exception.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (one per text).
        """
        try:
            return self.primary.embed_documents(texts)
        except Exception as e:
            logger.warning(
                "Primary embedding (%s) failed, falling back to %s: %s",
                self.primary_name,
                self.fallback_name,
                e,
            )
            return self.fallback.embed_documents(texts)

    def embed_query(self, text):
        """
        Generate an embedding for a single query string (sync).

        Tries the primary model first, falls back on any exception.

        Args:
            text: The query string to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        try:
            return self.primary.embed_query(text)
        except Exception as e:
            logger.warning(
                "Primary embedding (%s) failed, falling back to %s: %s",
                self.primary_name,
                self.fallback_name,
                e,
            )
            return self.fallback.embed_query(text)

    async def aembed_documents(self, texts):
        """
        Generate embeddings for a list of text documents (async).

        Tries the primary model first, falls back on any exception.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (one per text).
        """
        try:
            return await self.primary.aembed_documents(texts)
        except Exception as e:
            logger.warning(
                "Primary async embedding (%s) failed, falling back to %s: %s",
                self.primary_name,
                self.fallback_name,
                e,
            )
            return await self.fallback.aembed_documents(texts)

    async def aembed_query(self, text):
        """
        Generate an embedding for a single query string (async).

        Tries the primary model first, falls back on any exception.

        Args:
            text: The query string to embed.

        Returns:
            The embedding vector as a list of floats.
        """
        try:
            return await self.primary.aembed_query(text)
        except Exception as e:
            logger.warning(
                "Primary async embedding (%s) failed, falling back to %s: %s",
                self.primary_name,
                self.fallback_name,
                e,
            )
            return await self.fallback.aembed_query(text)


# ── LLM factory ───────────────────────────────────────────────────────────────


def get_llm(temperature=0.0, cache: bool | None = None):
    """
    Build and return a LangChain LLM with an automatic two-level fallback chain.

    Chain order: primary (fast) → fallback_1 (large) → fallback_2 (Gemini)

    Args:
        temperature: Sampling temperature for generation.
                     0.0 = deterministic, higher = more creative.
        cache: Controls LangChain's global LLM cache for this call chain.
               None  → use the global cache if one is configured (default).
               False → always bypass the cache (use for live/dynamic data).
               True  → force-enable the cache.

    Returns:
        A LangChain Runnable (ChatModel) with `.with_fallbacks()` applied.
    """
    # Only include `cache` in kwargs when explicitly set so that
    # None (use global default) is preserved rather than overriding it.
    extra_kwargs: dict = {}
    if cache is not None:
        extra_kwargs["cache"] = cache

    primary_llm = ChatGroq(
        model=LLM_FACTORY_CONFIG.primary_model,
        temperature=temperature,
        max_retries=LLM_FACTORY_CONFIG.max_retries,
        timeout=LLM_FACTORY_CONFIG.timeout,
        **extra_kwargs,
    )

    fallback_1_llm = ChatGroq(
        model=LLM_FACTORY_CONFIG.fallback_1_model,
        temperature=temperature,
        max_retries=LLM_FACTORY_CONFIG.max_retries,
        timeout=LLM_FACTORY_CONFIG.timeout,
        **extra_kwargs,
    )

    fallback_2_llm = ChatGoogleGenerativeAI(
        model=LLM_FACTORY_CONFIG.fallback_2_model,
        temperature=temperature,
        max_retries=LLM_FACTORY_CONFIG.max_retries,
        timeout=LLM_FACTORY_CONFIG.timeout,
        **extra_kwargs,
    )

    # Queue: primary → fallback_1 → fallback_2
    return primary_llm.with_fallbacks([fallback_1_llm, fallback_2_llm])


# ── Embedding factory ─────────────────────────────────────────────────────────


def get_embeddings(dimensions: int = 768):
    """
    Build and return a FallbackEmbeddings instance with configurable output dimensions.

    Primary:  OpenRouter text-embedding-3-small (via OPENROUTER_API_KEY)
    Fallback: Google Gemini Embedding (via GOOGLE_API_KEY)

    Args:
        dimensions: Output vector dimensionality.
                    768 (default) is used for FAQ / Qdrant search.
                    512 is used for the semantic FAQ response cache.

    Returns:
        A FallbackEmbeddings instance wrapping both models.
    """
    primary_name = LLM_FACTORY_CONFIG.primary_embedding_model
    fallback_name = LLM_FACTORY_CONFIG.fallback_embedding_model

    openrouter_embeddings = OpenAIEmbeddings(
        model=primary_name,
        dimensions=dimensions,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model=fallback_name, output_dimensionality=dimensions
    )

    return FallbackEmbeddings(
        primary=openrouter_embeddings,
        fallback=gemini_embeddings,
        primary_name=primary_name,
        fallback_name=fallback_name,
    )
