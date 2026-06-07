"""Shared LangGraph state definitions."""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State passed between graph nodes."""

    messages: Annotated[list, add_messages]
    customer_id: str
    session_id: str
    intent: str
    confidence: float
    query: str
    agent_response: dict[str, Any]
    structured_output: dict[str, Any]
    error: str | None
    final_response: dict[str, Any]
    next: str
    chat_summary: str
    summarized_message_count: int
    pending_agents: list[str]
    executed_agents: list[str]

    # Multimodal search fields
    image_base64: str | None
    image_azure_url: str | None
    image_tags: list[str] | None
    image_embedding: list[float] | None
    image_is_safe: bool | None
    visual_results: list[dict[str, Any]] | None
    active_filters: dict[str, Any] | None
