"""
Fetch all orders belonging to a customer, with optional status and date filtering.
"""

from typing import Annotated, Optional

from langchain_core.tools import InjectedToolArg, tool
from langfuse import observe

from app.db.supabase import get_supabase_client
from app.schemas.agent import ToolNotFoundResponse
from app.schemas.order import (
    CustomerOrdersResponse,
    OrderSummary,
)


@tool
@observe(name="tool_get_customer_orders")
async def get_customer_orders(
    customer_id: Annotated[str, InjectedToolArg],
    status: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """
    Retrieve orders for a customer with optional filters.

    Args:
        customer_id: Unique customer identifier (injected automatically).
        status: Optional order status filter (e.g. "placed", "processing", "shipped",
                "in_transit", "delivered", "cancelled").
        from_date: Optional ISO-8601 start date to filter orders placed on or
                   after this date (e.g. "2026-01-01").
        to_date: Optional ISO-8601 end date to filter orders placed on or
                 before this date (e.g. "2026-06-01").
        limit: Max number of orders to return (default 20).

    Returns:
        Customer orders matching the filters.
    """
    supabase = await get_supabase_client()

    query = supabase.table("orders").select("""
            id,
            status,
            payment_status,
            subtotal,
            shipping_cost,
            tax,
            total,
            ordered_at,
            tracking_number,
            carrier,
            estimated_delivery,
            delivered_at,
            return_eligible,
            return_deadline,
            payment,
            shipment,
            notes
            """).eq("customer_id", customer_id)

    if status:
        status = status.replace("-", "_").lower()
        query = query.eq("status", status)
    if from_date:
        query = query.gte("ordered_at", from_date)
    if to_date:
        query = query.lte("ordered_at", to_date)

    result = await query.order("ordered_at", desc=True).limit(limit).execute()

    if not result.data:
        response = ToolNotFoundResponse(
            message=(
                "I couldn't find any orders that match your current search criteria. "
                "This could be because no orders were placed during that specific timeframe, or perhaps none of your orders currently have that status. "
                "If you'd like, you can try adjusting the date range or status, and I'll gladly check again for you!"
            )
        )
        return response.model_dump(mode="json")

    orders = [
        OrderSummary(
            order_id=row["id"],
            status=row["status"],
            payment_status=row["payment_status"],
            subtotal=row.get("subtotal", 0),
            shipping_cost=row.get("shipping_cost", 0),
            tax=row.get("tax", 0),
            total_amount=row["total"],
            ordered_at=row["ordered_at"],
            tracking_number=row.get("tracking_number"),
            carrier=row.get("carrier"),
            estimated_delivery=row.get("estimated_delivery"),
            delivered_at=row.get("delivered_at"),
            return_eligible=row.get("return_eligible", False),
            return_deadline=row.get("return_deadline"),
            payment=row.get("payment"),
            shipment=row.get("shipment"),
            notes=row.get("notes"),
        )
        for row in result.data
    ]

    response = CustomerOrdersResponse(
        customer_id=customer_id,
        orders=orders,
    )

    return response.model_dump(mode="json")
