"""
Utility script to manually create and reset Postgres checkpoint tables for LangGraph.
"""

import os
import sys

import psycopg
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

# Adjust Python path to allow root imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.utils import setup_database_schema

# ── Private helpers ───────────────────────────────────────────────────────────


def _run_migrations(db_url: str) -> None:
    """
    Connect to the Postgres instance and execute LangGraph PostgresSaver migrations.

    Args:
        db_url: The database connection URL string.
    """
    with psycopg.connect(db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            # ── 1. Clear old migration state ──────────────────────────────────
            # We drop the migrations table first so LangGraph is forced to run setup cleanly.
            cur.execute("DROP TABLE IF EXISTS checkpoint_migrations;")
            print("Dropped old checkpoint_migrations table.")

            # ── 2. Run LangGraph migrations ───────────────────────────────────
            # Execute each of the exact create statements from PostgresSaver.
            for migration in PostgresSaver.MIGRATIONS:
                print(f"Executing migration...\n{migration.strip()[:50]}...")
                cur.execute(migration)


# ── Public main ───────────────────────────────────────────────────────────────


def main() -> None:
    """
    Connects to the Supabase Postgres instance via psycopg and forces the creation
    of LangGraph checkpoint tables by dropping any corrupted migration state.
    """
    load_dotenv()
    db_url = os.getenv("SUPABASE_DB_URL")

    if not db_url:
        print("Error: SUPABASE_DB_URL is not set.")
        return

    # ── 1. Apply general app schema ───────────────────────────────────────────
    setup_database_schema()

    # ── 2. Apply LangGraph checkpoint tables ──────────────────────────────────
    print("Connecting to Supabase to create checkpoint tables...")
    try:
        _run_migrations(db_url)
        print("All LangGraph checkpoint tables created successfully!")
    except Exception as e:
        print(f"Failed to create tables: {e}")


if __name__ == "__main__":
    main()
