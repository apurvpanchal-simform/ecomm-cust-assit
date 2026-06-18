"""
Pydantic schemas for order representations, items, and search responses.
"""

from datetime import datetime

from pydantic import BaseModel

# ── Summary (used by get_customer_orders / filter) ────────────────────


class OrderSummary(BaseModel):
    """Represents a summary of an order, including its total amount, item count, and current status, with validation and serialization provided by Pydantic's BaseModel."""

    order_id: str
    status: str
    payment_status: str
    subtotal: float
    shipping_cost: float
    tax: float
    total_amount: float
    ordered_at: datetime
    tracking_number: str | None = None
    carrier: str | None = None
    estimated_delivery: datetime | None = None
    delivered_at: datetime | None = None
    return_eligible: bool
    return_deadline: datetime | None = None
    payment: dict | None = None
    shipment: dict | None = None
    notes: str | None = None


class CustomerOrdersResponse(BaseModel):
    """Response model for customer orders, including order details and pagination metadata."""

    customer_id: str
    orders: list[OrderSummary]


# ── Full detail (used by get_order_details) ───────────────────────────


class OrderItem(BaseModel):
    """Single line-item inside the `items` JSONB array."""

    name: str | None = None
    product_id: str | None = None
    quantity: int | None = None
    price: float | None = None
    image: str | None = None
    variant: str | None = None
    color: str | None = None
    size: str | None = None


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
    """Response model for order item search queries, containing a list of matching OrderItem objects and pagination information."""

    customer_id: str
    keyword: str
    matches: list[OrderItemMatch]
