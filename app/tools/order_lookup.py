"""
Fetch all orders belonging to a customer.
"""

from langchain_core.tools import tool

from app.db.supabase import supabase
from app.schemas.order import (
    OrderSummary,
    CustomerOrdersResponse,
)

@tool
def get_customer_orders(customer_id: str) -> dict:
    """
    Retrieve all orders associated with a customer.

    Args:
        customer_id: Unique customer identifier

    Returns:
        List of customer orders
    """

    result = (
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
        .order("created_at", desc=True)
        .execute()
    )

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
            return_eligible=row["return_eligible"]
        )
        for row in result.data
    ]

    response = CustomerOrdersResponse(
        customer_id=customer_id,
        orders=orders,
    )

    return response.model_dump()