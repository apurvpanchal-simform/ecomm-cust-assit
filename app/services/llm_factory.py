import os
import logging
from dotenv import load_dotenv

load_dotenv()
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)


def estimate_tokens(texts) -> int:
    """Roughly estimate tokens (chars / 4)."""
    if isinstance(texts, str):
        return max(1, len(texts) // 4)
    return sum(max(1, len(t) // 4) for t in texts)


class FallbackEmbeddings:
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


def get_llm(temperature=0.0):
    """Returns OpenAI/Groq models with fallbacks as requested."""

    oss_20b_llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        temperature=temperature, 
        max_retries=2, 
        timeout=15.0
    )

    gpt4o_mini_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=temperature,
        max_retries=1,
        timeout=15.0,
    )

    oss_120b_llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=temperature,
        max_retries=2,
        timeout=15.0,
    )

    # Queue: primary (20b) -> fallback 1 (4o-mini) -> fallback 2 (120b)
    return oss_20b_llm.with_fallbacks([gpt4o_mini_llm, oss_120b_llm])


def get_embeddings():
    """Returns OpenAI text-embedding-3-large embeddings with Gemini fallback."""
    primary_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    fallback_name = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")

    primary_embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        dimensions=1024,
        api_key=os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY")),
        base_url="https://openrouter.ai/api/v1"
    )
    
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model=fallback_name, output_dimensionality=768
    )

    return FallbackEmbeddings(
        primary=primary_embeddings,
        fallback=gemini_embeddings,
        primary_name=primary_name,
        fallback_name=fallback_name,
    )
