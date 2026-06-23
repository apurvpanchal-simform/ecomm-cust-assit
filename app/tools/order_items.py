"""
Tool for searching across all of a customer's order line-items by keyword.

Useful when the user asks "did I ever order a laptop?" or "which order had
the blue jacket?" — it scans the JSONB `items` column across all their orders.
"""

from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from langfuse import observe

from app.db.supabase import get_supabase_client
from app.schemas.agent import ToolNotFoundResponse
from app.schemas.order import OrderItem, OrderItemMatch, OrderItemSearchResponse

# ── Helper ────────────────────────────────────────────────────────────────────


def _item_matches_keyword(item_data: dict, keyword_lower: str) -> bool:
    """
    Check whether a single order line-item matches the given keyword.

    Performs a case-insensitive substring search against the item's name,
    description, variant, color, and size fields.

    Args:
        item_data: Raw dict from the JSONB `items` column of an order row.
        keyword_lower: The search keyword, already lowercased by the caller.

    Returns:
        True if any field contains the keyword, False otherwise.
    """
    name = (item_data.get("name") or "").lower()
    description = (item_data.get("description") or "").lower()
    variant = (item_data.get("variant") or "").lower()
    color = (item_data.get("color") or "").lower()
    size = (item_data.get("size") or "").lower()

    return (
        keyword_lower in name
        or keyword_lower in description
        or keyword_lower in variant
        or keyword_lower in color
        or keyword_lower in size
    )


# ── Tool ──────────────────────────────────────────────────────────────────────


@tool
@observe(name="tool_search_order_items")
async def search_order_items(
    customer_id: Annotated[str, InjectedToolArg], keyword: str
) -> dict:
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
    supabase = await get_supabase_client()

    # ── 1. Fetch all orders for this customer ─────────────────────────────────
    # We only need the columns required to build OrderItemMatch objects.
    result = await (
        supabase.table("orders")
        .select("id, items, status, ordered_at, total")
        .eq("customer_id", customer_id)
        .order("ordered_at", desc=True)
        .execute()
    )

    # ── 2. Scan every line-item for keyword matches ───────────────────────────
    keyword_lower = keyword.lower()
    matches: list[OrderItemMatch] = []

    for row in result.data:
        raw_items = row.get("items") or []
        for item_data in raw_items:
            # Skip malformed entries that are not dicts
            if not isinstance(item_data, dict):
                continue

            if _item_matches_keyword(item_data, keyword_lower):
                matches.append(
                    OrderItemMatch(
                        order_id=row["id"],
                        order_status=row["status"],
                        ordered_at=row["ordered_at"],
                        item=OrderItem(**item_data),
                        order_total=row["total"],
                    )
                )

    # ── 3. Return results or a friendly not-found message ────────────────────
    if not matches:
        response = ToolNotFoundResponse(
            message=(
                f"I've looked through your past orders, but I couldn't find anything matching '{keyword}'. "
                "It might be listed under a slightly different name or spelling, or perhaps it was ordered from a different account. "
                "If you can think of another name for it, I'd be happy to search again!"
            )
        )
        return response.model_dump(mode="json")

    response = OrderItemSearchResponse(
        customer_id=customer_id,
        keyword=keyword,
        matches=matches,
    )

    return response.model_dump(mode="json")
