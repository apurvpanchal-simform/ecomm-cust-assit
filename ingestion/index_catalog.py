"""
Run once to index all product images into Qdrant.
Usage: python ingestion/index_catalog.py
"""

import asyncio
import os
import sys

import requests
from dotenv import load_dotenv
from qdrant_client.models import PointStruct

# Ensure app is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supabase import create_client

from app.db.qdrant import ensure_product_images_collection, get_qdrant_client
from ingestion.utils import setup_database_schema
from app.services.dense_embedder import embed_image_bytes
from app.services.sparse_embedder import embed_sparse_text
from app.services.storage_service import upload_product_image

load_dotenv(override=True)

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
qdrant = get_qdrant_client()


# ── Private helpers ───────────────────────────────────────────────────────────


def _fetch_products_list() -> list[dict]:
    """
    Fetch all product records from the Supabase database.

    Returns:
        A list of dictionaries representing products.
    """
    return supabase.table("products").select("*").execute().data


def _process_single_product(product: dict) -> PointStruct | None:
    """
    Process a single product by downloading its image, uploading to Azure,
    generating dense and sparse vectors, and building a Qdrant PointStruct.

    Also updates the product's azure_image_url in Supabase.

    Args:
        product: The product dictionary.

    Returns:
        PointStruct for indexing, or None if processing fails.
    """
    try:
        # 1. Download source image
        source_url = product.get("image")
        if not source_url:
            print(f"⚠️ No image URL for {product['id']}")
            return None

        img_bytes = requests.get(source_url).content

        # 2. Upload to Azure Blob Storage
        azure_url = upload_product_image(img_bytes, str(product["id"]))

        # 3. Generate Dense SigLIP embedding
        dense_vector = embed_image_bytes(img_bytes)

        # 4. Generate Sparse BM25 embedding
        text_to_embed = f"{product.get('title', '')} {product.get('category', '')} {product.get('description', '')}"
        sparse_vector = embed_sparse_text(text_to_embed)

        # 5. Update image_url in Supabase
        supabase.table("products").update({"azure_image_url": azure_url}).eq(
            "id", product["id"]
        ).execute()

        # 6. Build PointStruct
        return PointStruct(
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
    except Exception as e:
        print(f"❌ Failed to process product {product.get('id')}: {e}")
        return None


async def _upsert_points_in_batches(points: list[PointStruct]) -> None:
    """
    Upsert PointStruct vectors into the product_images collection in batches.

    Args:
        points: List of PointStruct objects to upsert.
    """
    if not points:
        print("No points to index.")
        return

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        await qdrant.upsert(collection_name="product_images", points=batch)
        print(f"✅ Indexed {min(i + batch_size, len(points))}/{len(points)}")


# ── Public Entrypoint ─────────────────────────────────────────────────────────


async def index_all_products() -> None:
    """
    Fetches all products from Supabase, generates multimodal dense and sparse embeddings,
    uploads images to Azure Blob Storage, and indexes the vectors and payloads into Qdrant.
    """
    # ── 1. Apply general app schema ───────────────────────────────────────────
    setup_database_schema()

    # ── 2. Initialize Qdrant Collection ───────────────────────────────────────
    # Recreate collection to match the new 1152 dimensions of SigLIP2
    await ensure_product_images_collection(recreate=True)

    # ── 3. Fetch products list ────────────────────────────────────────────────
    products = _fetch_products_list()
    if not products:
        print(
            "No products found in Supabase. Please populate the products table first."
        )
        return

    # ── 4. Process each product ───────────────────────────────────────────────
    points = []
    for product in products:
        point = _process_single_product(product)
        if point:
            points.append(point)

    # ── 5. Upsert to Qdrant ───────────────────────────────────────────────────
    await _upsert_points_in_batches(points)


if __name__ == "__main__":
    asyncio.run(index_all_products())
