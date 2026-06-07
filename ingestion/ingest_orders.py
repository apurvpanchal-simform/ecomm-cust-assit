import json
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def seed_orders():
    with open("data/orders.json", "r") as f:
        orders = json.load(f)

    batch_size = 100

    for i in range(0, len(orders), batch_size):
        batch = orders[i : i + batch_size]
        
        # Remove fields that exist in the legacy JSON but not in our Postgres schema
        for order in batch:
            order.pop("partition_key", None)
            order.pop("tracking_number", None)

        response = supabase.table("orders").upsert(batch, on_conflict="id").execute()

        print(f"Inserted/Updated {len(response.data)} orders")

    print(f"\nFinished seeding {len(orders)} orders.")


if __name__ == "__main__":
    seed_orders()
