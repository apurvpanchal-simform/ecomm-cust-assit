"""
Pydantic schemas for the human support agent API endpoints.
"""

from pydantic import BaseModel


class SupportReplyRequest(BaseModel):
    """Request body for a human support agent reply."""

    message: str
    agent_name: str


class TypingRequest(BaseModel):
    """Request body for broadcasting a support agent's typing indicator."""

    agent_name: str
