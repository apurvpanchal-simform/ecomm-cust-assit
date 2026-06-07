import os
import logging
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_nomic.embeddings import NomicEmbeddings

logger = logging.getLogger(__name__)

class FallbackEmbeddings:
    """A simple wrapper to fallback to a secondary embedding model on failure."""
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def embed_documents(self, texts):
        try:
            return self.primary.embed_documents(texts)
        except Exception as e:
            logger.warning(f"Primary embedding failed, using fallback: {e}")
            return self.fallback.embed_documents(texts)

    def embed_query(self, text):
        try:
            return self.primary.embed_query(text)
        except Exception as e:
            logger.warning(f"Primary embedding failed, using fallback: {e}")
            return self.fallback.embed_query(text)

    async def aembed_documents(self, texts):
        try:
            return await self.primary.aembed_documents(texts)
        except Exception as e:
            logger.warning(f"Primary async embedding failed, using fallback: {e}")
            return await self.fallback.aembed_documents(texts)

    async def aembed_query(self, text):
        try:
            return await self.primary.aembed_query(text)
        except Exception as e:
            logger.warning(f"Primary async embedding failed, using fallback: {e}")
            return await self.fallback.aembed_query(text)

def get_llm(temperature=0.0):
    """Returns Groq chat model with Google Gemini fallback."""
    # Primary: Groq
    groq_llm = ChatGroq(
        model=os.getenv("PRIMARY_MODEL", "llama-3.3-70b-versatile"), 
        temperature=temperature
    )
    
    # Fallback: Google Gemini
    gemini_llm = ChatGoogleGenerativeAI(
        model=os.getenv("FALLBACK_MODEL", "gemini-2.5-flash"),
        temperature=temperature
    )
    
    # LangChain fallback mechanism
    return groq_llm.with_fallbacks([gemini_llm])

def get_embeddings():
    """Returns Nomic embeddings with Gemini fallback (both matched to 768 dims)."""
    # Primary: Nomic (natively 768 dims)
    nomic_embeddings = NomicEmbeddings(model=os.getenv("NOMIC_EMBEDDING_MODEL", "nomic-embed-text-v1.5"))
    
    # Fallback: Gemini (explicitly configured to 768 dims to match Nomic)
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2"),
        output_dimensionality=768
    )
    
    return FallbackEmbeddings(primary=nomic_embeddings, fallback=gemini_embeddings)
