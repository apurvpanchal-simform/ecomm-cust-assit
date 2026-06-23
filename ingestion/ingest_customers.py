"""
Script to seed the Supabase database with sample customer records.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

from ingestion.utils import setup_database_schema

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# Initialize supabase client
client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ── Private helpers ───────────────────────────────────────────────────────────


def _load_customers_json() -> list[dict]:
    """
    Loads customer records from the data/customers.json file.

    Returns:
        A list of dictionaries representing customers.
    """
    file_path = Path(__file__).parent.parent / "data" / "customers.json"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _upsert_customers(customers: list[dict]) -> None:
    """
    Upsert customer records into the Supabase database.

    Args:
        customers: A list of customer dicts to insert or update.
    """
    response = client.table("customers").upsert(customers).execute()
    print(f"Successfully upserted {len(customers)} customers.")


# ── Public entrypoint ─────────────────────────────────────────────────────────


def ingest_customers() -> None:
    """
    Reads the customers JSON file and upserts the records into the Supabase database.
    """
    # ── 1. Initialize DB tables and schema ────────────────────────────────────
    setup_database_schema()

    # ── 2. Read customers data source ─────────────────────────────────────────
    customers = _load_customers_json()

    # ── 3. Upsert records to Supabase ─────────────────────────────────────────
    _upsert_customers(customers)


if __name__ == "__main__":
    ingest_customers()
