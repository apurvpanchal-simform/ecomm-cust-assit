import os
import logging
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_nomic.embeddings import NomicEmbeddings
from app.services.token_tracker import TokenCostCallbackHandler

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
        self.tracker = TokenCostCallbackHandler()

    def _log_cost(self, texts, model_name: str):
        tokens = estimate_tokens(texts)
        self.tracker.log_embedding_cost(model_name, tokens)

    def embed_documents(self, texts):
        try:
            res = self.primary.embed_documents(texts)
            self._log_cost(texts, self.primary_name)
            return res
        except Exception as e:
            logger.warning(f"Primary embedding failed, using fallback: {e}")
            res = self.fallback.embed_documents(texts)
            self._log_cost(texts, self.fallback_name)
            return res

    def embed_query(self, text):
        try:
            res = self.primary.embed_query(text)
            self._log_cost(text, self.primary_name)
            return res
        except Exception as e:
            logger.warning(f"Primary embedding failed, using fallback: {e}")
            res = self.fallback.embed_query(text)
            self._log_cost(text, self.fallback_name)
            return res

    async def aembed_documents(self, texts):
        try:
            res = await self.primary.aembed_documents(texts)
            self._log_cost(texts, self.primary_name)
            return res
        except Exception as e:
            logger.warning(f"Primary async embedding failed, using fallback: {e}")
            res = await self.fallback.aembed_documents(texts)
            self._log_cost(texts, self.fallback_name)
            return res

    async def aembed_query(self, text):
        try:
            res = await self.primary.aembed_query(text)
            self._log_cost(text, self.primary_name)
            return res
        except Exception as e:
            logger.warning(f"Primary async embedding failed, using fallback: {e}")
            res = await self.fallback.aembed_query(text)
            self._log_cost(text, self.fallback_name)
            return res

def get_llm(temperature=0.0):
    """Returns OpenAI GPT-4o with Groq fallback."""
    # Primary: OpenAI
    openai_llm = ChatOpenAI(
        model=os.getenv("PRIMARY_MODEL", "gpt-4o"),
        base_url='https://api.chatanywhere.tech/v1', 
        temperature=temperature,
        max_retries=0
    )
    
    # Fallback 1/3: Groq 20b
    groq_llm_1 = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=temperature,
        max_retries=0
    )
    
    # Fallback 2/4: Groq 120b
    groq_llm_2 = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=temperature,
        max_retries=0
    )
    
    # LangChain fallback mechanism (cyclic chain)
    return openai_llm.with_fallbacks([groq_llm_1, groq_llm_2, groq_llm_1, groq_llm_2])

def get_embeddings():
    """Returns Nomic embeddings with Gemini fallback (both matched to 768 dims)."""
    primary_name = os.getenv("NOMIC_EMBEDDING_MODEL", "nomic-embed-text-v1.5")
    fallback_name = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
    
    # Primary: Nomic (natively 768 dims)
    nomic_embeddings = NomicEmbeddings(model=primary_name)
    
    # Fallback: Gemini (explicitly configured to 768 dims to match Nomic)
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model=fallback_name,
        output_dimensionality=768
    )
    
    return FallbackEmbeddings(
        primary=nomic_embeddings, 
        fallback=gemini_embeddings,
        primary_name=primary_name,
        fallback_name=fallback_name
    )
