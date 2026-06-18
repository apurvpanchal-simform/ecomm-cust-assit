"""Response schemas for agent outputs and intent classification."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Structured output from support agents — enforced on every response."""

    resolution_text: str = Field(
        description="The final message to send to the user, combining any tool results."
    )


class ToolNotFoundResponse(BaseModel):
    """
    Standardized schema for returning 'not found' or error responses from tools.
    Provides human-in-the-loop feedback so the agent can help the user refine their query.
    """

    message: Optional[str] = Field(
        default=None,
        description="A helpful message explaining why no results were found, and suggesting possible reasons.",
    )
    error: Optional[str] = Field(
        default=None,
        description="An error message if the lookup failed or the entity was explicitly not found.",
    )


class Route(BaseModel):
    """Represents a route with origin and destination points. Inherits from BaseModel to provide validation and serialization."""

    pending_agents: list[Literal["faq", "order", "image_search_agent"]] = Field(
        description="The ordered list of agents to execute. Put data-gathering agents first if others depend on them. Return an empty list if the query is ONLY a greeting, chitchat, vague search, or completely out-of-domain."
    )
    sub_queries: dict[str, str] = Field(
        default_factory=dict,
        description="A dictionary mapping each pending agent name to its specific instructions (e.g., {'order': 'Find the last order', 'faq': 'Find the warranty for the item the order agent finds'}).",
    )
    response: str = Field(
        default="",
        description="If you cannot fulfill part or all of the query (due to safety, out-of-domain, or vagueness), or if it's a greeting, provide your direct response/refusal here.",
    )
    image_intent: bool = Field(
        default=False,
        description="Set to true if the user uploaded an image and wants to search or identify it.",
    )
    search_query: Optional[str] = Field(
        default=None,
        description="Extract a clear search query if finding products. Combine user's intent and image description. Do NOT include price terms.",
    )
    min_price: Optional[float] = Field(
        default=None,
        description="If the user mentions a minimum budget (e.g. 'over $20'), set the minimum price.",
    )
    max_price: Optional[float] = Field(
        default=None,
        description="If the user mentions a maximum budget (e.g. 'under $50', 'cheaper than 100'), set the maximum price.",
    )
