"""Support ticket data models for Cosmos DB storage."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from app.schemas.response import TicketCategory


class TicketStatus(StrEnum):
    """Support ticket lifecycle statuses."""

    AUTO_RESOLVED = "auto_resolved"
    PENDING_REVIEW = "pending_review"
    ESCALATED = "escalated"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"


class SupportTicket(BaseModel):
    """Full support ticket document — stored in Cosmos DB."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    customer_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = None
    category: TicketCategory
    resolution_text: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    requires_human: bool = False
    sentiment_trajectory: list[float] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    conversation_history: list[dict] = Field(default_factory=list)
    status: TicketStatus = TicketStatus.PENDING_REVIEW
    order_id: str | None = None
    assigned_agent: str | None = None
    notes: str | None = None
    partition_key: str = ""  # Same as customer_id

    def model_post_init(self, __context) -> None:
        """Auto-set partition_key from customer_id."""
        if not self.partition_key:
            self.partition_key = self.customer_id


class TicketCreate(BaseModel):
    """Input model for creating a new support ticket."""

    customer_id: str
    category: TicketCategory
    resolution_text: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    requires_human: bool = False
    sources: list[str] = Field(default_factory=list)
    sentiment_trajectory: list[float] = Field(default_factory=list)
    conversation_history: list[dict] = Field(default_factory=list)
    order_id: str | None = None


class TicketUpdate(BaseModel):
    """Partial update model for modifying a support ticket."""

    status: TicketStatus | None = None
    assigned_agent: str | None = None
    notes: str | None = None
    resolution_text: str | None = None
    requires_human: bool | None = None