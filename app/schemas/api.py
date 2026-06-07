from pydantic import BaseModel
from typing import List
from datetime import datetime

class LoginRequest(BaseModel):
    email: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    image_base64: str | None = None

class ConversationItem(BaseModel):
    conversation_id: str
    title: str
    updated_at: datetime
