"""Response schemas for agent outputs and intent classification."""

from enum import StrEnum

from pydantic import BaseModel, Field

from typing import Literal

from app.schemas.product import ProductCard


class MasterIntent(StrEnum):
    """Top-level intent categories for the master supervisor."""

    RECOMMENDATION = "recommendation"
    SUPPORT = "support"


class TicketCategory(StrEnum):
    """Support ticket categories — used by the support supervisor."""

    FAQ = "faq"
    ORDER = "order"
    ESCALATION = "escalation"
    RECOMMENDATION = "recommendation"


class IntentClassification(BaseModel):
    """Structured output from the master supervisor — classifies top-level intent."""

    intent: MasterIntent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Brief explanation of why this intent was chosen")


MasterIntentClassification = IntentClassification


class SupportIntentClassification(BaseModel):
    """Structured output from the support supervisor — classifies support sub-intent."""

    intent: TicketCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Brief explanation of the support category")

class RouteDecision(BaseModel):
    route: Literal["faq", "order"]
    confidence: float

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