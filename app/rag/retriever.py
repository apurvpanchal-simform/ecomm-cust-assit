import os
from dotenv import load_dotenv
from app.db.qdrant import get_qdrant_client, ensure_faq_collection
from app.services.llm_factory import get_embeddings

load_dotenv()

class FAQRetriever:
    def __init__(self):
        self.collection_name = "ecommerce-knowledge"
        self.client = get_qdrant_client()
        self.embeddings = get_embeddings()

    async def initialize(self, recreate: bool = False):
        await ensure_faq_collection(recreate=recreate)

    async def vector_search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search the vector store and return results with scores.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            A list of dicts, each containing 'content', 'source_file', and 'score'.
        """
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
