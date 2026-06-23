"""
Tool for fetching full details of a single order — items, pricing,
payment method, shipping status, and return eligibility.

Called by the Order agent when the user asks about a specific order ID.
"""

from datetime import datetime, timezone
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from langfuse import observe

from app.db.supabase import get_supabase_client
from app.schemas.agent import ToolNotFoundResponse
from app.schemas.order import OrderDetail, OrderItem

# ── Helper ────────────────────────────────────────────────────────────────────


def _calculate_days_remaining(
    deadline_str: str | None, return_eligible: bool
) -> int | None:
    """
    Compute how many days remain until the return deadline.

    Args:
        deadline_str: ISO-8601 string for the return deadline (e.g. "2026-07-01T00:00:00Z"),
                      or None if no deadline is set.
        return_eligible: Whether the order is currently eligible for a return.

    Returns:
        Number of days remaining (minimum 0) if eligible and a valid deadline exists,
        otherwise None.
    """
    if not deadline_str or not return_eligible:
        return None

    try:
        deadline = datetime.fromisoformat(deadline_str)
        now = datetime.now(timezone.utc)
        delta = deadline - now
        return max(0, delta.days)
    except (ValueError, TypeError):
        # Malformed deadline string — treat as unknown
        return None


# ── Tool ──────────────────────────────────────────────────────────────────────


@tool
@observe(name="tool_get_order_details")
async def get_order_details(
    customer_id: Annotated[str, InjectedToolArg], order_id: str
) -> dict:
    """
    Retrieve COMPLETE details for a specific order.

    Use this for ANY specific question about a single order, including:
    - What was ordered (items)
    - Pricing, taxes, and shipping costs
    - Payment method and status
    - Shipping status, carrier, tracking number, and delivery dates
    - Return eligibility and return deadline

    Args:
        customer_id: Unique customer identifier (injected automatically).
        order_id: The order ID to look up.

    Returns:
        Order detail with items, pricing, payment, shipping, and return info.
    """
    supabase = await get_supabase_client()

    # ── 1. Fetch order row from Supabase ─────────────────────────────────────
    # Filter by both order_id AND customer_id to prevent cross-customer data leaks.
    result = await (
        supabase.table("orders")
        .select("*")
        .eq("id", order_id)
        .eq("customer_id", customer_id)
        .limit(1)
        .execute()
    )

    if not result.data:
        response = ToolNotFoundResponse(
            error=(
                f"Hmm, I couldn't find any order matching the ID '{order_id}'. "
                "It's possible there might be a small typo in the order number, or the order was placed under a different account. "
                "Could you please double-check the order ID and let me know? I'd be happy to try again!"
            )
        )
        return response.model_dump(mode="json")

    row = result.data[0]

    # ── 2. Parse line items ───────────────────────────────────────────────────
    raw_items = row.get("items") or []
    items = [
        OrderItem(**item) if isinstance(item, dict) else OrderItem(name=str(item))
        for item in raw_items
    ]

    # ── 3. Compute return deadline countdown ─────────────────────────────────
    days_remaining = _calculate_days_remaining(
        deadline_str=row.get("return_deadline"),
        return_eligible=row.get("return_eligible", False),
    )

    # ── 4. Build and return the structured detail object ─────────────────────
    detail = OrderDetail(
        order_id=row["id"],
        customer_id=row["customer_id"],
        items=items,
        subtotal=row["subtotal"],
        shipping_cost=row.get("shipping_cost", 0),
        tax=row.get("tax", 0),
        total=row["total"],
        status=row["status"],
        payment_status=row["payment_status"],
        payment=row.get("payment"),
        shipment=row.get("shipment"),
        tracking_number=row.get("tracking_number"),
        carrier=row.get("carrier"),
        ordered_at=row["ordered_at"],
        estimated_delivery=row.get("estimated_delivery"),
        delivered_at=row.get("delivered_at"),
        return_eligible=row.get("return_eligible", False),
        return_deadline=row.get("return_deadline"),
        days_remaining=days_remaining,
        notes=row.get("notes"),
    )

    return detail.model_dump(mode="json")
