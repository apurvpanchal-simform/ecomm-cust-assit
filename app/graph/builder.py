from langgraph.graph import StateGraph, START, END
from app.agents.faq import faq_node
from app.agents.order import order_node
from app.agents.supervisor import supervisor_node, out_of_domain_node
from app.agents.summarizer import summarizer_node
from app.agents.clip_embedding import clip_embedding_node
from app.agents.visual_search import visual_search_node
from app.agents.result_formatter import result_formatter_node
from app.agents.image_analyzer import image_analyzer_node
from app.agents.filter_extractor import filter_extractor_node
from app.graph.state import AgentState


def route_supervisor(state: AgentState) -> str:
    """Returns the next node to execute from the state."""
    next_node = state.get("next", "FINISH")
    if next_node == "FINISH":
        return "summarizer"
    return next_node

builder = StateGraph(AgentState)

builder.add_node("image_analyzer", image_analyzer_node)
builder.add_node("supervisor", supervisor_node)
builder.add_node("faq", faq_node)
builder.add_node("order", order_node)
builder.add_node("out_of_domain", out_of_domain_node)
builder.add_node("filter_extractor", filter_extractor_node)
builder.add_node("clip_embedder", clip_embedding_node)
builder.add_node("visual_search", visual_search_node)
builder.add_node("result_formatter", result_formatter_node)
builder.add_node("summarizer", summarizer_node)

builder.add_edge(START, "image_analyzer")
builder.add_edge("image_analyzer", "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "faq": "faq",
        "order": "order",
        "visual_search_agent": "filter_extractor",
        "out_of_domain": "out_of_domain",
        "summarizer": "summarizer",
    },
)

def route_after_agent(state: AgentState) -> str:
    """Routes to the next unexecuted agent in pending_agents, or to summarizer if done."""
    pending = state.get("pending_agents", []) or []
    executed = state.get("executed_agents", []) or []
    
    for agent in pending:
        if agent not in executed:
            if agent == "visual_search_agent":
                return "filter_extractor"
            return agent
            
    return "summarizer"


# Register agent transitions using route_after_agent
builder.add_conditional_edges(
    "faq",
    route_after_agent,
    {
        "faq": "faq",
        "order": "order",
        "filter_extractor": "filter_extractor",
        "out_of_domain": "out_of_domain",
        "summarizer": "summarizer",
    }
)

builder.add_conditional_edges(
    "order",
    route_after_agent,
    {
        "faq": "faq",
        "order": "order",
        "filter_extractor": "filter_extractor",
        "out_of_domain": "out_of_domain",
        "summarizer": "summarizer",
    }
)

builder.add_conditional_edges(
    "out_of_domain",
    route_after_agent,
    {
        "faq": "faq",
        "order": "order",
        "filter_extractor": "filter_extractor",
        "out_of_domain": "out_of_domain",
        "summarizer": "summarizer",
    }
)

# Visual Search Pipeline
builder.add_edge("filter_extractor", "clip_embedder")
builder.add_edge("clip_embedder", "visual_search")
builder.add_edge("visual_search", "result_formatter")
builder.add_edge("result_formatter", "summarizer")

builder.add_edge("summarizer", END)

def compile_graph(checkpointer=None):
    """Compiles and returns the graph, optionally attaching a checkpointer."""
    return builder.compile(checkpointer=checkpointer)

