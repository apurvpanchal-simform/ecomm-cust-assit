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

client = create_client(SUPABASE_URL, SUPABASE_KEY)


def load_customers():
    """Loads customer records from data/customers.json."""
    file_path = Path(__file__).parent.parent / "data" / "customers.json"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ingest_customers():
    """
    Reads the customers JSON file and upserts the records into the Supabase database.
    """
    setup_database_schema()
    customers = load_customers()
    response = client.table("customers").upsert(customers).execute()
    print(f"Inserted/Updated {len(customers)} customers")
    return response


if __name__ == "__main__":
    ingest_customers()
