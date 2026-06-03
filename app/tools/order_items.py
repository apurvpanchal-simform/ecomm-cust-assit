"""
Search across all order items for a customer by keyword.
"""

from langchain_core.tools import tool

from app.db.supabase import get_supabase_client
from app.schemas.order import OrderItem, OrderItemMatch, OrderItemSearchResponse


@tool
def search_order_items(customer_id: str, keyword: str) -> dict:
    """
    Search through all of a customer's orders to find items matching a keyword.

    Performs a case-insensitive match against item names and descriptions
    inside the JSONB `items` column.

    Use this when the customer asks things like "did I ever order a laptop?"
    or "which order had the red shoes?".

    Args:
        customer_id: Unique customer identifier (injected automatically).
        keyword: Product name or keyword to search for (e.g. "AirPods",
                 "blue jacket", "laptop").

    Returns:
        Matching items across all orders.
    """
    supabase = get_supabase_client()

    result = (
        supabase.table("orders")
        .select("id, items, status, ordered_at, total")
        .eq("customer_id", customer_id)
        .order("ordered_at", desc=True)
        .execute()
    )

    keyword_lower = keyword.lower()
    matches: list[OrderItemMatch] = []

    for row in result.data:
        raw_items = row.get("items") or []
        for item_data in raw_items:
            if not isinstance(item_data, dict):
                continue

            name = (item_data.get("name") or "").lower()
            description = (item_data.get("description") or "").lower()
            variant = (item_data.get("variant") or "").lower()

            if keyword_lower in name or keyword_lower in description or keyword_lower in variant:
                matches.append(
                    OrderItemMatch(
                        order_id=row["id"],
                        order_status=row["status"],
                        ordered_at=row["ordered_at"],
                        item=OrderItem(**item_data),
                        order_total=row["total"],
                    )
                )

    response = OrderItemSearchResponse(
        customer_id=customer_id,
        keyword=keyword,
        matches=matches,
    )

    return response.model_dump(mode="json")
