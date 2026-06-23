"""
Script to seed the Supabase database with the initial product catalog.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.utils import setup_database_schema

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ── Private helpers ───────────────────────────────────────────────────────────


def _load_products_json() -> list[dict]:
    """
    Loads product records from the data/products.json file.

    Returns:
        A list of dictionaries representing products.
    """
    file_path = Path(__file__).parent.parent / "data" / "products.json"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _upsert_products_in_batches(products: list[dict], batch_size: int = 100) -> None:
    """
    Upsert product records in batches into the Supabase database.

    Args:
        products: A list of product dicts to insert or update.
        batch_size: Number of records to process per batch.
    """
    for i in range(0, len(products), batch_size):
        batch = products[i : i + batch_size]
        response = supabase.table("products").upsert(batch).execute()
        print(f"Inserted/Updated {len(response.data)} products")


# ── Public entrypoint ─────────────────────────────────────────────────────────


def ingest_products() -> None:
    """
    Reads the products JSON file and upserts them in batches into the Supabase database.
    """
    # ── 1. Initialize database tables ─────────────────────────────────────────
    setup_database_schema()

    # ── 2. Read products from local JSON ──────────────────────────────────────
    products = _load_products_json()

    # ── 3. Upsert products in batches ─────────────────────────────────────────
    _upsert_products_in_batches(products, batch_size=100)
    print(f"Finished seeding {len(products)} products.")


if __name__ == "__main__":
    ingest_products()
