"""Shared LangGraph state definitions for the customer assistant graph.

`AgentState` is a TypedDict that is threaded through every node in the graph.
LangGraph merges each node's return dict into this state using the reducer
functions declared via `Annotated` (e.g. `add_messages` for the messages list).

Field groups:
    Core          — query, customer identity, routing, errors
    Conversation  — message history, rolling summary, summarized count
    Multimodal    — image data, embeddings, safety state
    Search        — image search results, active filters, sub-queries
    FAQ           — retrieved knowledge chunks
"""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Shared state passed between all graph nodes.

    `total=False` means every field is optional, so individual nodes only need
    to return the keys they actually modify.

    Fields annotated with `Annotated[list, add_messages]` use LangGraph's
    built-in message reducer which appends new messages rather than replacing
    the whole list.
    """

    # ── Core routing state ────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]  # Full conversation message history
    customer_id: str  # Authenticated customer UUID
    query: str  # Raw text of the latest user message
    error: str | None  # Last error string, or None
    next: str  # Next node the supervisor routes to

    # ── Conversation memory ───────────────────────────────────────────────────
    chat_summary: str  # Rolling markdown summary of older turns
    summarized_message_count: int  # Index up to which messages are summarized

    # ── Multi-agent routing ───────────────────────────────────────────────────
    pending_agents: list[str]  # Agents the supervisor queued for this turn
    executed_agents: list[str]  # Agents that have already run this turn

    # ── Escalation ────────────────────────────────────────────────────────────
    escalate_to_human: bool | None  # True → bypass AI and route to human support

    # ── Multimodal / image pipeline ───────────────────────────────────────────
    image_base64: str | None  # Raw base64-encoded uploaded image
    image_tags: list[str] | None  # Object tags extracted by Azure CV
    image_embedding: list[float] | None  # SigLIP2 dense vector for the image
    image_is_safe: bool | None  # True if Azure CV found no unsafe tags
    image_results: list[dict[str, Any]] | None  # Final ranked product results
    active_filters: dict[str, Any] | None  # Price filters extracted by the supervisor
    image_description: str | None  # Azure CV caption + OCR text
    search_query: str | None  # Clean product search query for image search
    sub_queries: dict[str, str] | None  # Per-agent task strings set by supervisor

    # ── FAQ specific ──────────────────────────────────────────────────────────
    faq_chunks: list[dict[str, Any]] | None  # Raw Qdrant chunks returned by search_faq

    # ── Ephemeral safety state (reset each turn) ──────────────────────────────
    # Set by image_analyzer_node when an unsafe image is detected.
    # Cleared by cleanup_node at the end of the visual search pipeline.
    image_safety_warning: str | None
