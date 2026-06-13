"""
Pydantic schemas for the FastAPI endpoints.
"""

from datetime import datetime

from pydantic import BaseModel


class LoginRequest(BaseModel):
    """Schema for a customer login request."""

    email: str


class LoginResponse(BaseModel):
    """Schema for a successful login response."""

    access_token: str
    token_type: str = "bearer"


class ChatRequest(BaseModel):
    """Schema for an incoming chat message request."""

    query: str
    conversation_id: str | None = None
    image_base64: str | None = None


class ConversationItem(BaseModel):
    """Schema representing a single conversation history metadata item."""

    conversation_id: str
    title: str
    updated_at: datetime
