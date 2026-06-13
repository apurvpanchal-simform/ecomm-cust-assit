"""
Utility script to manually create and reset Postgres checkpoint tables for LangGraph.
"""

import os

import psycopg
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver


def main():
    """
    Connects to the Supabase Postgres instance via psycopg and forces the creation
    of LangGraph checkpoint tables by dropping any corrupted migration state.
    """
    load_dotenv()
    db_url = os.getenv("SUPABASE_DB_URL")

    if not db_url:
        print("Error: SUPABASE_DB_URL is not set.")
        return

    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.utils import setup_database_schema

    setup_database_schema()

    print("Connecting to Supabase to create checkpoint tables...")

    # We connect directly using psycopg
    try:
        with psycopg.connect(db_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                # 1. To fix any broken migration state, we drop the migrations table first
                # so that LangGraph is forced to re-run the setup cleanly.
                cur.execute("DROP TABLE IF EXISTS checkpoint_migrations;")
                print("Dropped old checkpoint_migrations table.")

                # 2. We can also manually execute the exact create table statements
                # just to be absolutely certain they exist.
                for migration in PostgresSaver.MIGRATIONS:
                    print(f"Executing migration...\n{migration.strip()[:50]}...")
                    cur.execute(migration)

                print("All LangGraph checkpoint tables created successfully!")
    except Exception as e:
        print(f"Failed to create tables: {e}")


if __name__ == "__main__":
    main()
