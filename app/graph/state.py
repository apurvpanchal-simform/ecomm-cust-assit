from typing import TypedDict, Optional, List

from app.schemas.response import AgentResponse


class AgentState(TypedDict, total=False):
    query: str

    customer_id: str
    order_id: str

    intent: str

    retrieved_chunks: List[dict]

    support_response: AgentResponse