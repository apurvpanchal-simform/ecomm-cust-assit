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


def load_products():
    """Loads product records from data/products.json."""
    file_path = Path(__file__).parent.parent / "data" / "products.json"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ingest_products():
    """
    Reads the products JSON file and upserts them in batches into the Supabase database.
    """
    setup_database_schema()
    products = load_products()

    batch_size = 100
    for i in range(0, len(products), batch_size):
        batch = products[i : i + batch_size]
        response = supabase.table("products").upsert(batch).execute()
        print(f"Inserted/Updated {len(response.data)} products")

    print(f"Finished seeding {len(products)} products.")


if __name__ == "__main__":
    ingest_products()
