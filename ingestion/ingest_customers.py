import json
import os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

client = create_client(SUPABASE_URL, SUPABASE_KEY)

def create_table_if_not_exists():
    """Ensure the customers table exists before ingestion."""
    if not SUPABASE_DB_URL:
        print("Warning: SUPABASE_DB_URL not set. Skipping table creation.")
        return
        
    try:
        conn = psycopg2.connect(SUPABASE_DB_URL)
        conn.autocommit = True
        cursor = conn.cursor()
        
        create_sql = """
        CREATE TABLE IF NOT EXISTS customers (
            id TEXT PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            email TEXT,
            phone TEXT,
            address JSONB,
            loyalty_tier TEXT,
            created_at TIMESTAMP WITH TIME ZONE
        );
        """
        cursor.execute(create_sql)
        cursor.execute("NOTIFY pgrst, 'reload schema'")
        cursor.close()
        conn.close()
        print("Table 'customers' ensured and schema cache reloaded.")
    except Exception as e:
        print(f"Database setup error: {e}")

def load_customers():
    file_path = Path(__file__).parent.parent / "data" / "customers.json"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def ingest_customers():
    create_table_if_not_exists()
    customers = load_customers()
    response = client.table("customers").upsert(customers).execute()
    print(f"Inserted/Updated {len(customers)} customers")
    return response

if __name__ == "__main__":
    ingest_customers()
