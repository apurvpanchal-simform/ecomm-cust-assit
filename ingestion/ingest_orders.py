import json
import os
from datetime import datetime, timezone, timedelta
from dateutil import parser
import psycopg2
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def create_table_if_not_exists():
    """Ensure the orders table exists before ingestion."""
    if not SUPABASE_DB_URL:
        print("Warning: SUPABASE_DB_URL not set. Skipping table creation.")
        return

    try:
        conn = psycopg2.connect(SUPABASE_DB_URL)
        conn.autocommit = True
        cursor = conn.cursor()

        create_sql = """
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            customer_id TEXT,
            items JSONB,
            subtotal NUMERIC,
            shipping_cost NUMERIC,
            tax NUMERIC,
            total NUMERIC,
            status TEXT,
            payment_status TEXT,
            payment JSONB,
            carrier TEXT,
            shipment JSONB,
            ordered_at TIMESTAMP WITH TIME ZONE,
            estimated_delivery TIMESTAMP WITH TIME ZONE,
            delivered_at TIMESTAMP WITH TIME ZONE,
            return_eligible BOOLEAN,
            return_deadline TIMESTAMP WITH TIME ZONE,
            notes TEXT
        );
        """
        cursor.execute(create_sql)
        cursor.execute("NOTIFY pgrst, 'reload schema'")
        cursor.close()
        conn.close()
        print("Table 'orders' ensured and schema cache reloaded.")
    except Exception as e:
        print(f"Database setup error: {e}")


def shift_timestamps(orders):
    """Dynamically shift orders so the latest one is exactly 1 day ago."""
    max_date = None
    for order in orders:
        dt = parser.parse(order["ordered_at"])
        if max_date is None or dt > max_date:
            max_date = dt

    now = datetime.now(timezone.utc)
    offset = now - max_date - timedelta(days=1)

    def shift_date_str(date_str):
        if not date_str:
            return date_str
        new_dt = parser.parse(date_str) + offset
        return new_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    for order in orders:
        if "ordered_at" in order:
            order["ordered_at"] = shift_date_str(order["ordered_at"])
        if "estimated_delivery" in order:
            order["estimated_delivery"] = shift_date_str(order["estimated_delivery"])
        if "delivered_at" in order:
            order["delivered_at"] = shift_date_str(order["delivered_at"])
        if "return_deadline" in order:
            order["return_deadline"] = shift_date_str(order["return_deadline"])
        if "shipment" in order:
            if "estimated_delivery" in order["shipment"]:
                order["shipment"]["estimated_delivery"] = shift_date_str(
                    order["shipment"]["estimated_delivery"]
                )
            if "delivered_at" in order["shipment"]:
                order["shipment"]["delivered_at"] = shift_date_str(
                    order["shipment"]["delivered_at"]
                )
    return orders


def seed_orders():
    create_table_if_not_exists()

    # Fetch product images to attach to order items
    products_response = (
        supabase.table("products").select("id, azure_image_url, image").execute()
    )
    product_map = {
        str(p["id"]): p.get("azure_image_url") or p.get("image")
        for p in products_response.data
    }

    with open("data/orders.json", "r") as f:
        orders = json.load(f)

    # Automatically make timestamps fresh relative to today
    orders = shift_timestamps(orders)

    batch_size = 100
    for i in range(0, len(orders), batch_size):
        batch = orders[i : i + batch_size]
        for order in batch:
            order.pop("partition_key", None)
            order.pop("tracking_number", None)

            for item in order.get("items", []):
                pid = str(item.get("product_id"))
                if pid in product_map and product_map[pid]:
                    item["image"] = product_map[pid]

        response = supabase.table("orders").upsert(batch, on_conflict="id").execute()
        print(f"Inserted/Updated {len(response.data)} orders")

    print(f"\\nFinished seeding {len(orders)} orders.")


if __name__ == "__main__":
    seed_orders()
