from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    image_base64: str | None = None