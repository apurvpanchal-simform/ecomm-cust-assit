"""
FAQ knowledge base retriever backed by Qdrant vector search.

`FAQRetriever` wraps the Qdrant client and the LangChain embedding model so
callers only need to call `vector_search(query, top_k)` to get ranked FAQ chunks.
"""

from dotenv import load_dotenv

from app.config import SEARCH_CONFIG
from app.db.qdrant import ensure_faq_collection, get_qdrant_client
from app.services.llm_factory import get_embeddings

load_dotenv()


class FAQRetriever:
    """
    Retrieves FAQ chunks from a Qdrant vector collection using semantic search.

    Embeds the query with the configured text embedding model and runs a
    nearest-neighbour search against the `ecommerce-knowledge` collection.

    Typical flow:
        retriever = FAQRetriever()
        chunks = await retriever.vector_search(query="what is your return policy?", top_k=5)
        # → list of dicts with 'content', 'source_file', and 'score' keys
    """

    def __init__(self):
        self.collection_name = "ecommerce-knowledge"
        self.client = get_qdrant_client()
        # Uses the default 768-dimension embedding model (OpenRouter with Gemini fallback)
        self.embeddings = get_embeddings()

    async def initialize(self, recreate: bool = False) -> None:
        """
        Ensure the Qdrant collection exists, optionally recreating it.

        Should be called during ingestion pipelines that need to set up the
        collection from scratch.  Not needed during normal serving — the
        collection is expected to already exist at startup.

        Args:
            recreate: If True, drop and recreate the collection.
                      If False (default), only create it if it does not exist.
        """
        await ensure_faq_collection(recreate=recreate)

    async def vector_search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Embed `query` and return the top-k most similar FAQ chunks from Qdrant.

        Args:
            query: The natural-language search query (e.g. "how long do returns take?").
            top_k: Maximum number of results to return (default is read from SEARCH_CONFIG.faq_top_k).

        Returns:
            A list of dicts, each containing:
                content     (str)   — raw text of the FAQ chunk.
                source_file (str)   — the source document filename.
                score       (float) — cosine similarity score (higher = more relevant).
        """
        if top_k is None:
            top_k = SEARCH_CONFIG.faq_top_k

        # Embed the query into the same vector space as the indexed FAQ documents
        query_vector = await self.embeddings.aembed_query(query)

        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )

        return [
            {
                "content": hit.payload.get("content", ""),
                "source_file": hit.payload.get("source_file", "unknown"),
                "score": hit.score,
            }
            for hit in results.points
        ]
