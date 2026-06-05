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
