"""Order data models for e-commerce order tracking and management."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class OrderStatus(StrEnum):
    """Order lifecycle statuses."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    RETURNED = "returned"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PaymentStatus(StrEnum):
    """Payment lifecycle statuses."""

    PENDING = "pending"
    AUTHORIZED = "authorized"
    PAID = "paid"
    FAILED = "failed"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"


class OrderItem(BaseModel):
    """A single item within an order."""

    product_id: str
    name: str
    quantity: int = Field(ge=1)
    price: float = Field(ge=0)
    size: str | None = None
    color: str | None = None


class ShippingAddress(BaseModel):
    """Customer shipping address."""

    street: str
    city: str
    state: str
    zip_code: str
    country: str = "US"


class Payment(BaseModel):
    """Payment details safe for internal support workflows."""

    status: PaymentStatus = PaymentStatus.PAID
    method: str | None = None
    transaction_id: str | None = None
    refunded_amount: float = Field(default=0.0, ge=0)


class Shipment(BaseModel):
    """Shipment and tracking information for an order."""

    tracking_number: str | None = None
    carrier: str | None = None
    status: OrderStatus = OrderStatus.PROCESSING
    estimated_delivery: datetime | None = None
    delivered_at: datetime | None = None


class Order(BaseModel):
    """Full order document — mirrors the Cosmos DB schema."""

    id: str
    customer_id: str
    items: list[OrderItem]
    subtotal: float = Field(ge=0)
    shipping_cost: float = Field(ge=0, default=0.0)
    tax: float = Field(ge=0, default=0.0)
    total: float = Field(ge=0)
    status: OrderStatus = OrderStatus.PENDING
    payment_status: PaymentStatus = PaymentStatus.PAID
    payment: Payment | None = None
    tracking_number: str | None = None
    carrier: str | None = None
    shipment: Shipment | None = None
    shipping_address: ShippingAddress | None = None
    ordered_at: datetime
    estimated_delivery: datetime | None = None
    delivered_at: datetime | None = None
    return_eligible: bool = True
    return_deadline: datetime | None = None
    notes: str | None = None
    partition_key: str = ""  # Same as customer_id for Cosmos DB

    def model_post_init(self, __context) -> None:
        """Auto-set partition_key from customer_id."""
        if not self.partition_key:
            self.partition_key = self.customer_id


class OrderHistory(BaseModel):
    """Aggregated order history for a customer."""

    customer_id: str
    orders: list[Order]
    total_orders: int = 0
    total_spent: float = 0.0

    def model_post_init(self, __context) -> None:
        """Compute aggregates from orders list."""
        if not self.total_orders:
            self.total_orders = len(self.orders)
        if not self.total_spent:
            self.total_spent = sum(o.total for o in self.orders)


class ReturnEligibility(BaseModel):
    """Return eligibility result for order support flows."""

    order_id: str
    eligible: bool
    reason: str
    policy_source: str
    required_next_actions: list[str] = Field(default_factory=list)
    return_deadline: datetime | None = None