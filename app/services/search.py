import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

class VectorStore:
    def __init__(self):
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = "ecommerce-knowledge"
        self.client = QdrantClient(url=self.url)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL")
        )

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )

    def initialize(self):
        self._ensure_collection()

    def vector_search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search the vector store and return results with scores.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            A list of dicts, each containing 'content', 'source_file', and 'score'.
        """
        query_vector = self.embeddings.embed_query(query)

        results = self.client.query_points(
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
