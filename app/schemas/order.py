from typing import Optional, Literal, Any
from pydantic import BaseModel
from datetime import datetime

class OrderReference(BaseModel):
    reference_type: Literal[
        "explicit_order_id",
        "latest_order",
        "previous_order",
        "conversation_reference"
    ]

    order_id: Optional[str] = None

class OrderContextResult(BaseModel):
    success: bool
    order_id: Optional[str] = None
    order: Optional[dict] = None
    error: Optional[str] = None

class OrderSummary(BaseModel):
    order_id: str
    status: str
    payment_status: str
    total_amount: float
    ordered_at: datetime
    carrier: str | None = None
    estimated_delivery: datetime | None = None
    delivered_at: datetime | None = None
    return_eligible: bool

class CustomerOrdersResponse(BaseModel):
    customer_id: str
    orders: list[OrderSummary]