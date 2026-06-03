"""Response schemas for agent outputs and intent classification."""

from enum import StrEnum

from pydantic import BaseModel, Field

class TicketCategory(StrEnum):
    """Support ticket categories — used by the support supervisor."""

    FAQ = "faq"
    ORDER = "order"
    ESCALATION = "escalation"
    RECOMMENDATION = "recommendation"
class AgentResponse(BaseModel):
    """Structured output from support agents — enforced on every response."""

    resolution_text: str = Field(description="The customer-facing response text")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Agent's confidence in the response (0-1)"
    )
    ticket_category: TicketCategory = Field(
        description="Classified category of the support interaction"
    )
    requires_human: bool = Field(default=False, description="Whether this needs human agent review")
    sources: list[str] = Field(
        default_factory=list, description="Document chunks used to generate the response"
    )
    suggested_actions: list[str] = Field(default_factory=list, description="Recommended next steps")
    escalation_reason: str | None = Field(
        default=None, description="Why this was escalated (if applicable)"
    )
    order_id: str | None = Field(
        default=None, description="Related order ID (if order-related query)"
    )