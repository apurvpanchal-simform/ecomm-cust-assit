"""
Utility functions for database schema initialization during ingestion.
"""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")


# ── Private helpers ───────────────────────────────────────────────────────────


def _load_schema_sql() -> str:
    """
    Read and return the contents of the database schema SQL file.

    Returns:
        The content of app/db/schema.sql as a string.
    """
    schema_path = Path(__file__).parent.parent / "app" / "db" / "schema.sql"
    with open(schema_path, "r", encoding="utf-8") as f:
        return f.read()


def _execute_schema_sql(create_sql: str) -> None:
    """
    Connect to the Supabase database and execute the schema SQL script.

    Args:
        create_sql: The raw SQL commands to construct the database schema.
    """
    if not SUPABASE_DB_URL:
        raise ValueError("SUPABASE_DB_URL is not set.")

    with psycopg.connect(SUPABASE_DB_URL, autocommit=True) as conn:
        with conn.cursor() as cursor:
            # Execute the entire schema script
            cursor.execute(create_sql)
            # Reload PostgREST schema cache to recognize new tables/views immediately
            cursor.execute("NOTIFY pgrst, 'reload schema'")


# ── Public function ───────────────────────────────────────────────────────────


def setup_database_schema() -> None:
    """Reads app/db/schema.sql and executes it to create all tables and indexes."""
    if not SUPABASE_DB_URL:
        print("Warning: SUPABASE_DB_URL not set. Skipping table creation.")
        return

    try:
        # ── 1. Load schema SQL script ─────────────────────────────────────────
        create_sql = _load_schema_sql()

        # ── 2. Connect and apply schema to the database ───────────────────────
        _execute_schema_sql(create_sql)
        print("Schema successfully applied from schema.sql.")
    except Exception as e:
        print(f"Database setup error: {e}")
