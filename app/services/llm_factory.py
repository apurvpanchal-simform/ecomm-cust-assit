"""
Factory functions for generating LLM and Embedding models with fallback chains.
"""

import logging
import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)


def estimate_tokens(texts) -> int:
    """Roughly estimate tokens (chars / 4)."""
    if isinstance(texts, str):
        return max(1, len(texts) // 4)
    return sum(max(1, len(t) // 4) for t in texts)


class FallbackEmbeddings(Embeddings):
    """A simple wrapper to fallback to a secondary embedding model on failure."""

    def __init__(self, primary, fallback, primary_name: str, fallback_name: str):
        self.primary = primary
        self.fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name

    def embed_documents(self, texts):
        try:
            res = self.primary.embed_documents(texts)
            return res
        except Exception as e:
            logger.warning(f"Primary embedding failed, using fallback: {e}")
            res = self.fallback.embed_documents(texts)
            return res

    def embed_query(self, text):
        try:
            res = self.primary.embed_query(text)
            return res
        except Exception as e:
            logger.warning(f"Primary embedding failed, using fallback: {e}")
            res = self.fallback.embed_query(text)
            return res

    async def aembed_documents(self, texts):
        try:
            res = await self.primary.aembed_documents(texts)
            return res
        except Exception as e:
            logger.warning(f"Primary async embedding failed, using fallback: {e}")
            res = await self.fallback.aembed_documents(texts)
            return res

    async def aembed_query(self, text):
        try:
            res = await self.primary.aembed_query(text)
            return res
        except Exception as e:
            logger.warning(f"Primary async embedding failed, using fallback: {e}")
            res = await self.fallback.aembed_query(text)
            return res


def get_llm(temperature=0.0, cache: bool | None = None):
    """Returns Groq models with fallbacks as requested.

    Args:
        temperature: Sampling temperature for generation.
        cache: Set to False to disable the global LLM cache for this call chain,
               None to use the global cache if set, or True to force it.
    """

    # Build kwargs dict; only include cache when explicitly set so that
    # the default (None → use global cache if configured) is preserved.
    extra_kwargs: dict = {}
    if cache is not None:
        extra_kwargs["cache"] = cache

    primary_llm = ChatGroq(
        model="openai/gpt-oss-20b", temperature=temperature, max_retries=2, timeout=15.0,
        **extra_kwargs,
    )

    fallback_1_llm = ChatGroq(
        model="openai/gpt-oss-120b", temperature=temperature, max_retries=2, timeout=15.0,
        **extra_kwargs,
    )

    fallback_2_llm = ChatGroq(
        model="groq-compound",
        temperature=temperature,
        max_retries=2,
        timeout=15.0,
        **extra_kwargs,
    )

    # Queue: primary -> fallback 1 -> fallback 2
    return primary_llm.with_fallbacks([fallback_1_llm, fallback_2_llm])


def get_embeddings():
    """Returns OpenRouter text-embedding-3-small embeddings with Gemini fallback."""
    primary_name = os.getenv(
        "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
    )
    fallback_name = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")

    openrouter_embeddings = OpenAIEmbeddings(
        model=primary_name,
        dimensions=768,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model=fallback_name, output_dimensionality=768
    )

    return FallbackEmbeddings(
        primary=openrouter_embeddings,
        fallback=gemini_embeddings,
        primary_name=primary_name,
        fallback_name=fallback_name,
    )
