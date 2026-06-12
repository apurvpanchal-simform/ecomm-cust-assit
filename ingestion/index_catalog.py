"""
Run once to index all product images into Qdrant.
Usage: python ingestion/index_catalog.py
"""

import requests
from qdrant_client.models import PointStruct
import asyncio
import os
import sys
import uuid
from dotenv import load_dotenv

# Ensure app is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.qdrant import get_qdrant_client, ensure_product_images_collection
from app.services.dense_embedder import embed_image_bytes
from app.services.sparse_embedder import embed_sparse_text

# from app.services.vision_service import vectorize_image_bytes
from app.services.storage_service import upload_product_image
from supabase import create_client

load_dotenv(override=True)

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
qdrant = get_qdrant_client()


async def index_all_products():
    # Recreate collection to match the new 1152 dimensions of SigLIP2
    await ensure_product_images_collection(recreate=True)

    # Fetch all products from Supabase
    products = supabase.table("products").select("*").execute().data
    points = []

    if not products:
        print(
            "No products found in Supabase. Please populate the products table first."
        )
        return

    for product in products:
        try:
            # 1. Download source image
            source_url = product.get("image")
            if not source_url:
                print(f"⚠️ No image URL for {product['id']}")
                continue

            img_bytes = requests.get(source_url).content

            # 2. Upload to Azure Blob Storage
            azure_url = upload_product_image(img_bytes, str(product["id"]))

            # 3. Generate Dense SigLIP embedding
            dense_vector = embed_image_bytes(img_bytes)

            # 4. Generate Sparse BM25 embedding
            text_to_embed = f"{product.get('title', '')} {product.get('category', '')} {product.get('description', '')}"
            sparse_vector = embed_sparse_text(text_to_embed)

            points.append(
                PointStruct(
                    id=product["id"],
                    vector={"": dense_vector, "text": sparse_vector},
                    payload={
                        "product_id": product["id"],
                        "title": product["title"],
                        "price": float(product["price"]),
                        "image_url": azure_url,
                        "category": product.get("category", ""),
                        "description": product.get("description", ""),
                    },
                )
            )

            # Update image_url in Supabase
            supabase.table("products").update({"azure_image_url": azure_url}).eq(
                "id", product["id"]
            ).execute()

        except Exception as e:
            print(f"❌ Failed {product['id']}: {e}")

    # Batch upsert into Qdrant (100 at a time)
    if points:
        for i in range(0, len(points), 100):
            await qdrant.upsert(
                collection_name="product_images", points=points[i : i + 100]
            )
            print(f"✅ Indexed {min(i+100, len(points))}/{len(points)}")
    else:
        print("No points to index.")


if __name__ == "__main__":
    asyncio.run(index_all_products())
