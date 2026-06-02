from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    customer_id: str