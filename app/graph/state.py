"""Shared LangGraph state definitions and state utility helpers."""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State passed between graph nodes."""

    messages: Annotated[list, add_messages]
    customer_id: str
    session_id: str
    query: str
    error: str | None
    next: str
    chat_summary: str
    summarized_message_count: int
    pending_agents: list[str]
    executed_agents: list[str]
    escalate_to_human: bool | None


    # Multimodal search fields
    image_base64: str | None
    image_azure_url: str | None
    image_tags: list[str] | None
    image_embedding: list[float] | None
    image_is_safe: bool | None
    image_results: list[dict[str, Any]] | None
    active_filters: dict[str, Any] | None
    image_description: str | None
    search_query: str | None
    sub_queries: dict[str, str] | None
    # FAQ specific fields
    faq_chunks: list[dict[str, Any]] | None
    # Ephemeral safety warning set by image_analyzer (reset each turn)
    image_safety_warning: str | None
