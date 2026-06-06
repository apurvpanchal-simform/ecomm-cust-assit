import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

def create_product_images_collection():
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    
    qdrant = QdrantClient(url=qdrant_url)
    
    collection_name = "product_images"
    
    try:
        # Check if collection exists
        qdrant.get_collection(collection_name)
        print(f"Collection {collection_name} already exists.")
    except Exception:
        # Create collection if it doesn't exist
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=512,           # CLIP ViT-B/32 output dim
                distance=Distance.COSINE
            )
        )
        # Index payload fields for filtered search
        qdrant.create_payload_index(collection_name, "price", PayloadSchemaType.FLOAT)
        print(f"Collection {collection_name} created.")

if __name__ == "__main__":
    create_product_images_collection()
