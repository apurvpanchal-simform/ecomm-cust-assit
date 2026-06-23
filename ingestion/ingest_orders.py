"""
Script to seed the Supabase database with sample orders.
Dynamically shifts timestamps to make the data appear fresh.
"""

import json
import os
from datetime import datetime, timedelta, timezone

from dateutil import parser
from dotenv import load_dotenv
from supabase import Client, create_client

from ingestion.utils import setup_database_schema

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ── Private Helpers for Timestamp Shifting ────────────────────────────────────


def _find_max_date(orders: list[dict]) -> datetime:
    """
    Find the most recent (maximum) order timestamp in the dataset.

    Args:
        orders: List of raw order dictionaries.

    Returns:
        The latest datetime found across all orders.
    """
    max_date = None
    for order in orders:
        dt = parser.parse(order["ordered_at"])
        if max_date is None or dt > max_date:
            max_date = dt
    return max_date


def _shift_date_str(date_str: str | None, offset: timedelta) -> str | None:
    """
    Apply a time offset to an ISO-8601 date string.

    Args:
        date_str: The ISO date string to shift.
        offset: The timedelta to shift by.

    Returns:
        The shifted ISO-8601 date string, or None if the input was empty.
    """
    if not date_str:
        return date_str
    new_dt = parser.parse(date_str) + offset
    return new_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _apply_shifts_to_order(order: dict, offset: timedelta) -> None:
    """
    In-place adjust all date fields within an order by the given offset.

    Args:
        order: The order dictionary to update.
        offset: The timedelta to shift by.
    """
    if "ordered_at" in order:
        order["ordered_at"] = _shift_date_str(order["ordered_at"], offset)
    if "estimated_delivery" in order:
        order["estimated_delivery"] = _shift_date_str(
            order["estimated_delivery"], offset
        )
    if "delivered_at" in order:
        order["delivered_at"] = _shift_date_str(order["delivered_at"], offset)
    if "return_deadline" in order:
        order["return_deadline"] = _shift_date_str(order["return_deadline"], offset)
    if "shipment" in order:
        ship = order["shipment"]
        if "estimated_delivery" in ship:
            ship["estimated_delivery"] = _shift_date_str(
                ship["estimated_delivery"], offset
            )
        if "delivered_at" in ship:
            ship["delivered_at"] = _shift_date_str(ship["delivered_at"], offset)


def shift_timestamps(orders: list[dict]) -> list[dict]:
    """
    Dynamically shift orders so the latest one is exactly 1 day ago.

    Args:
        orders: The list of raw order records.

    Returns:
        The modified orders list with shifted timestamps.
    """
    max_date = _find_max_date(orders)
    now = datetime.now(timezone.utc)
    offset = now - max_date - timedelta(days=1)

    for order in orders:
        _apply_shifts_to_order(order, offset)

    return orders


# ── Private Helpers for Order Seeding ─────────────────────────────────────────


def _get_product_image_map() -> dict[str, str]:
    """
    Fetch all products from Supabase and build a map of product_id to image URL.

    Returns:
        A dictionary mapping product ID strings to their image URLs.
    """
    products_response = (
        supabase.table("products").select("id, azure_image_url, image").execute()
    )
    return {
        str(p["id"]): p.get("azure_image_url") or p.get("image")
        for p in products_response.data
    }


def _load_orders_json() -> list[dict]:
    """
    Loads order records from the data/orders.json file.

    Returns:
        A list of dictionaries representing orders.
    """
    with open("data/orders.json", "r") as f:
        return json.load(f)


def _preprocess_and_enrich_batch(
    batch: list[dict], product_map: dict[str, str]
) -> None:
    """
    Clean up partitions and attach image URLs to items in the batch.

    Args:
        batch: A subset list of order dicts.
        product_map: Map of product ID strings to image URLs.
    """
    for order in batch:
        order.pop("partition_key", None)

        for item in order.get("items", []):
            pid = str(item.get("product_id"))
            if pid in product_map and product_map[pid]:
                item["image"] = product_map[pid]


# ── Public Entrypoint ─────────────────────────────────────────────────────────


def seed_orders() -> None:
    """
    Reads orders.json, shifts all dates so the latest order is exactly 1 day old,
    attaches product images to order items, and upserts them into Supabase.
    """
    # ── 1. Apply general app schema ───────────────────────────────────────────
    setup_database_schema()

    # ── 2. Load context maps and orders source ────────────────────────────────
    product_map = _get_product_image_map()
    orders = _load_orders_json()

    # ── 3. Apply timestamp shifting ───────────────────────────────────────────
    orders = shift_timestamps(orders)

    # ── 4. Process and upload in batches ──────────────────────────────────────
    batch_size = 100
    for i in range(0, len(orders), batch_size):
        batch = orders[i : i + batch_size]
        _preprocess_and_enrich_batch(batch, product_map)

        response = supabase.table("orders").upsert(batch, on_conflict="id").execute()
        print(f"Inserted/Updated {len(response.data)} orders")

    print(f"\nFinished seeding {len(orders)} orders.")


if __name__ == "__main__":
    seed_orders()
