"""Response schemas for agent outputs and intent classification."""
from enum import StrEnum
from typing import Optional
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Structured output from support agents — enforced on every response."""

    resolution_text: str = Field(description="The final message to send to the user, combining any tool results.")


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
