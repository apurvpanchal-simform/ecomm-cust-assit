from typing import Optional, Literal, Any
from pydantic import BaseModel
from datetime import datetime

# ── Reference helpers (existing) ──────────────────────────────────────


class OrderReference(BaseModel):
    reference_type: Literal[
        "explicit_order_id", "latest_order", "previous_order", "conversation_reference"
    ]

    order_id: Optional[str] = None


class OrderContextResult(BaseModel):
    success: bool
    order_id: Optional[str] = None
    order: Optional[dict] = None
    error: Optional[str] = None


# ── Summary (used by get_customer_orders / filter) ────────────────────


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


# ── Full detail (used by get_order_details) ───────────────────────────


class OrderItem(BaseModel):
    """Single line-item inside the `items` JSONB array."""

    name: str | None = None
    product_id: str | None = None
    quantity: int | None = None
    price: float | None = None
    image_url: str | None = None
    variant: str | None = None


class OrderDetail(BaseModel):
    """Complete representation of a single order including items, pricing, payment, shipping, and returns."""

    order_id: str
    customer_id: str
    items: list[OrderItem]
    subtotal: float
    shipping_cost: float
    tax: float
    total: float
    status: str
    payment_status: str
    payment: dict | None = None
    shipment: dict | None = None
    tracking_number: str | None = None
    carrier: str | None = None
    ordered_at: datetime
    estimated_delivery: datetime | None = None
    delivered_at: datetime | None = None
    return_eligible: bool
    return_deadline: datetime | None = None
    days_remaining: int | None = None
    notes: str | None = None


# ── Item search (used by search_order_items) ──────────────────────────


class OrderItemMatch(BaseModel):
    """A matched item found inside an order."""

    order_id: str
    order_status: str
    ordered_at: datetime
    item: OrderItem
    order_total: float


class OrderItemSearchResponse(BaseModel):
    customer_id: str
    keyword: str
    matches: list[OrderItemMatch]
