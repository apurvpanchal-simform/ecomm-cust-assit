import os
from functools import lru_cache
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

@lru_cache(maxsize=1)
def get_qdrant_client() -> AsyncQdrantClient:
    """Return a singleton AsyncQdrantClient."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", None)
    return AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)

async def ensure_faq_collection():
    """Ensure the FAQ knowledge collection exists."""
    client = get_qdrant_client()
    collection_name = "ecommerce-knowledge"
    try:
        await client.get_collection(collection_name)
    except Exception:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

async def ensure_product_images_collection():
    """Ensure the product images collection exists."""
    client = get_qdrant_client()
    collection_name = "product_images"
    try:
        await client.get_collection(collection_name)
    except Exception:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=512,           # CLIP ViT-B/32 output dim
                distance=Distance.COSINE
            )
        )
        # Index payload fields for filtered search
        await client.create_payload_index(collection_name, "price", PayloadSchemaType.FLOAT)
