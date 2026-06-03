"""
Fetch all orders belonging to a customer, with optional status and date filtering.
"""

from typing import Optional
from langsmith import traceable

from langchain_core.tools import tool

from app.db.supabase import get_supabase_client
from app.schemas.order import (
    OrderSummary,
    CustomerOrdersResponse,
)


@tool
@traceable(name="tool_get_customer_orders")
def get_customer_orders(
    customer_id: str,
    status: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """
    Retrieve orders for a customer with optional filters.

    Args:
        customer_id: Unique customer identifier (injected automatically).
        status: Optional order status filter (e.g. "delivered", "shipped",
                "processing", "cancelled").
        from_date: Optional ISO-8601 start date to filter orders placed on or
                   after this date (e.g. "2026-01-01").
        to_date: Optional ISO-8601 end date to filter orders placed on or
                 before this date (e.g. "2026-06-01").
        limit: Max number of orders to return (default 20).

    Returns:
        Customer orders matching the filters.
    """
    supabase = get_supabase_client()

    query = (
        supabase.table("orders")
        .select(
            """
            id,
            status,
            payment_status,
            total,
            ordered_at,
            carrier,
            estimated_delivery,
            delivered_at,
            return_eligible
            """
        )
        .eq("customer_id", customer_id)
    )

    if status:
        query = query.eq("status", status)
    if from_date:
        query = query.gte("ordered_at", from_date)
    if to_date:
        query = query.lte("ordered_at", to_date)

    result = query.order("ordered_at", desc=True).limit(limit).execute()

    orders = [
        OrderSummary(
            order_id=row["id"],
            status=row["status"],
            payment_status=row["payment_status"],
            total_amount=row["total"],
            ordered_at=row["ordered_at"],
            carrier=row["carrier"],
            estimated_delivery=row["estimated_delivery"],
            delivered_at=row["delivered_at"],
            return_eligible=row["return_eligible"],
        )
        for row in result.data
    ]

    response = CustomerOrdersResponse(
        customer_id=customer_id,
        orders=orders,
    )

    return response.model_dump(mode="json")