import json
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

client = create_client(SUPABASE_URL, SUPABASE_KEY)


def load_customers():

    file_path = Path(__file__).parent.parent / "data" / "customers.json"

    with open(file_path, "r", encoding="utf-8") as f:

        return json.load(f)


def ingest_customers():

    customers = load_customers()

    response = client.table("customers").upsert(customers).execute()

    print(f"Inserted/Updated " f"{len(customers)} customers")

    return response


if __name__ == "__main__":
    ingest_customers()
