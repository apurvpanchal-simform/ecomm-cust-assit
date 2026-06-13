"""
Utility functions for database schema initialization during ingestion.
"""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")


def setup_database_schema():
    """Reads app/db/schema.sql and executes it to create all tables and indexes."""
    if not SUPABASE_DB_URL:
        print("Warning: SUPABASE_DB_URL not set. Skipping table creation.")
        return

    try:
        schema_path = Path(__file__).parent.parent / "app" / "db" / "schema.sql"
        with open(schema_path, "r", encoding="utf-8") as f:
            create_sql = f.read()

        conn = psycopg.connect(SUPABASE_DB_URL, autocommit=True)
        cursor = conn.cursor()

        # Execute the entire schema
        cursor.execute(create_sql)
        cursor.execute("NOTIFY pgrst, 'reload schema'")

        cursor.close()
        conn.close()
        print("Schema successfully applied from schema.sql.")
    except Exception as e:
        print(f"Database setup error: {e}")
