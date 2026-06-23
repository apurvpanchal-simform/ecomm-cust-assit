"""
LangGraph graph builder module.

Defines all graph nodes, their wiring (edges), and conditional routing
functions, then exposes `compile_graph()` to produce the final runnable.

Graph topology overview
-----------------------
START
  ├─ (image present?) → image_analyzer → supervisor
  └─ (no image)       → supervisor

supervisor
  ├─ faq            → (route_after_agent) → synthesizer | more agents
  ├─ order          → (route_after_agent) → synthesizer | more agents
  └─ image_search_agent → clip_embedder → image_search → cleanup
                                                              └─ (route_after_agent)

synthesizer → summarizer → END
summarizer  → END          (when supervisor routes directly to FINISH)
"""

from langgraph.graph import END, START, StateGraph

from app.agents.clip_embedding import clip_embedding_node
from app.agents.faq import faq_node
from app.agents.image_analyzer import image_analyzer_node
from app.agents.image_search import image_search_node
from app.agents.order import order_node
from app.agents.summarizer import summarizer_node
from app.agents.supervisor import supervisor_node
from app.agents.synthesizer import synthesizer_node
from app.graph.cleanup import cleanup_node
from app.graph.state import AgentState

# ── Routing functions ─────────────────────────────────────────────────────────


def route_start(state: AgentState) -> str:
    """
    Entry-point router: decides whether to pre-process an uploaded image first.

    If the user included an image in their message, we run `image_analyzer`
    before the supervisor so the supervisor has access to tags and OCR text.

    Returns:
        "image_analyzer" if an image is present, otherwise "supervisor".
    """
    if state.get("image_base64"):
        return "image_analyzer"
    return "supervisor"


def route_supervisor(state: AgentState) -> str:
    """
    Post-supervisor router: read `state.next` and forward to the correct node.

    When the supervisor sets `next = "FINISH"` (e.g. for chitchat or refusals),
    we skip all agents and go directly to the summarizer.

    Returns:
        The name of the next node to execute.
    """
    next_node = state.get("next", "FINISH")
    if next_node == "FINISH":
        return "summarizer"
    return next_node


def route_after_agent(state: AgentState) -> str:
    """
    Post-agent router: check if there are more pending agents to run.

    Iterates through `pending_agents` and returns the first agent that has not
    yet appeared in `executed_agents`.  If all pending agents are done, routes
    to the summarizer (if 1 or 0 agents executed) or the synthesizer (if multiple
    agents executed).

    The image search pipeline is treated as an alias: "image_search_agent" in
    `pending_agents` maps to the "clip_embedder" node (the pipeline entry point).

    Returns:
        The name of the next node to execute.
    """
    pending = state.get("pending_agents", []) or []
    executed = state.get("executed_agents", []) or []

    for agent in pending:
        if agent not in executed:
            # Image search starts at clip_embedder, not image_search directly
            if agent == "image_search_agent":
                return "clip_embedder"
            return agent

    if len(pending) <= 1:
        return "summarizer"
    return "synthesizer"


# ── Node registration ─────────────────────────────────────────────────────────

builder = StateGraph(AgentState)

# Pre-processing nodes (run before the supervisor)
builder.add_node("image_analyzer", image_analyzer_node)

# Routing node (decides which agents to invoke)
builder.add_node("supervisor", supervisor_node)

# Specialist agent nodes
builder.add_node("faq", faq_node)
builder.add_node("order", order_node)

# Image search pipeline nodes (run sequentially after routing)
builder.add_node("clip_embedder", clip_embedding_node)  # Step 1: generate embedding
builder.add_node("image_search", image_search_node)  # Step 2: Qdrant hybrid search
builder.add_node("cleanup", cleanup_node)  # Step 3: clear heavy state fields

# Post-processing nodes
builder.add_node("synthesizer", synthesizer_node)  # Merge multiple agent responses
builder.add_node("summarizer", summarizer_node)  # Update rolling conversation summary


# ── Edge wiring ───────────────────────────────────────────────────────────────

# Entry point: route to image_analyzer or directly to supervisor
builder.add_conditional_edges(
    START,
    route_start,
    {
        "image_analyzer": "image_analyzer",
        "supervisor": "supervisor",
    },
)

# image_analyzer always feeds directly into supervisor
builder.add_edge("image_analyzer", "supervisor")

# Supervisor routes to the first pending agent (or to summarizer on FINISH)
builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "faq": "faq",
        "order": "order",
        "image_search_agent": "clip_embedder",
        "synthesizer": "synthesizer",
        "summarizer": "summarizer",
    },
)

# After faq or order runs, check if more agents need to run
AGENT_ROUTES = ["faq", "order", "clip_embedder", "synthesizer", "summarizer"]
builder.add_conditional_edges("faq", route_after_agent, AGENT_ROUTES)
builder.add_conditional_edges("order", route_after_agent, AGENT_ROUTES)

# Image search pipeline runs as a fixed sequence
builder.add_edge("clip_embedder", "image_search")
builder.add_edge("image_search", "cleanup")
# After cleanup, check if more agents (e.g. faq) also need to run
builder.add_conditional_edges("cleanup", route_after_agent, AGENT_ROUTES)

# Final steps: synthesizer → summarizer → END
builder.add_edge("synthesizer", "summarizer")
builder.add_edge("summarizer", END)


# ── Graph compilation ─────────────────────────────────────────────────────────


def compile_graph(checkpointer=None):
    """
    Compile and return the assembled LangGraph execution pipeline.

    Args:
        checkpointer: Optional checkpoint saver (e.g. `AsyncDualCheckpointer`)
                      for persisting conversation state across turns.

    Returns:
        The compiled LangGraph runnable application.
    """
    return builder.compile(checkpointer=checkpointer)
