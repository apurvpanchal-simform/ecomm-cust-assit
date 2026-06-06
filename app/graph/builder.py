from langgraph.graph import StateGraph, START, END
from app.agents.faq import faq_node
from app.agents.order import order_node
from app.agents.supervisor import supervisor_node, out_of_domain_node
from app.agents.summarizer import summarizer_node
from app.agents.content_moderation import content_moderation_node
from app.agents.clip_embedder_node import clip_embedding_node
from app.agents.visual_search import visual_search_node
from app.agents.result_formatter import result_formatter_node
from app.graph.state import AgentState


def route_supervisor(state: AgentState) -> str:
    """Returns the next node to execute from the state."""
    next_node = state.get("next", "FINISH")
    if next_node == "FINISH":
        return "summarizer"
    return next_node

def route_content_moderation(state: AgentState) -> str:
    """If image is unsafe, skip to summarizer."""
    if state.get("image_is_safe") is False:
        return "summarizer"
    return "clip_embedder"


builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("faq", faq_node)
builder.add_node("order", order_node)
builder.add_node("out_of_domain", out_of_domain_node)
builder.add_node("content_moderation", content_moderation_node)
builder.add_node("clip_embedder", clip_embedding_node)
builder.add_node("visual_search", visual_search_node)
builder.add_node("result_formatter", result_formatter_node)
builder.add_node("summarizer", summarizer_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "faq": "faq",
        "order": "order",
        "visual_search_agent": "content_moderation",
        "out_of_domain": "out_of_domain",
        "summarizer": "summarizer",
    },
)

builder.add_edge("faq", "supervisor")
builder.add_edge("order", "supervisor")

# Visual Search Pipeline
builder.add_conditional_edges(
    "content_moderation",
    route_content_moderation,
    {
        "summarizer": "summarizer",
        "clip_embedder": "clip_embedder",
    }
)
builder.add_edge("clip_embedder", "visual_search")
builder.add_edge("visual_search", "result_formatter")
builder.add_edge("result_formatter", "summarizer")

builder.add_edge("out_of_domain", "summarizer")
builder.add_edge("summarizer", END)


def compile_graph(checkpointer=None):
    """Compiles and returns the graph, optionally attaching a checkpointer."""
    return builder.compile(checkpointer=checkpointer)

