from langgraph.graph import StateGraph, START, END
from app.agents.faq import faq_node
from app.agents.order import order_node
from app.agents.supervisor import supervisor_node
from app.agents.summarizer import summarizer_node
from app.agents.synthesizer import synthesizer_node
from app.agents.clip_embedding import clip_embedding_node
from app.agents.visual_search import visual_search_node

from app.agents.image_analyzer import image_analyzer_node
from app.agents.filter_extractor import filter_extractor_node
from app.agents.cleanup import cleanup_node
from app.graph.state import AgentState

def route_start(state: AgentState) -> str:
    """Routes to image_analyzer if an image is present, else directly to supervisor."""
    if state.get("image_base64"):
        return "image_analyzer"
    return "supervisor"

def route_supervisor(state: AgentState) -> str:
    """Returns the next node to execute from the state."""
    next_node = state.get("next", "FINISH")
    if next_node == "FINISH":
        return END
    return next_node

builder = StateGraph(AgentState)

builder.add_node("image_analyzer", image_analyzer_node)
builder.add_node("supervisor", supervisor_node)
builder.add_node("faq", faq_node)
builder.add_node("order", order_node)
builder.add_node("filter_extractor", filter_extractor_node)
builder.add_node("clip_embedder", clip_embedding_node)
builder.add_node("visual_search", visual_search_node)

builder.add_node("cleanup", cleanup_node)
builder.add_node("synthesizer", synthesizer_node)
builder.add_node("summarizer", summarizer_node)

builder.add_conditional_edges(
    START,
    route_start,
    {
        "image_analyzer": "image_analyzer",
        "supervisor": "supervisor",
    }
)
builder.add_edge("image_analyzer", "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "faq": "faq",
        "order": "order",
        "visual_search_agent": "filter_extractor",
        "synthesizer": "synthesizer",
        END: END,
    },
)

def route_after_agent(state: AgentState) -> str:
    """Routes to the next unexecuted agent in pending_agents, or to synthesizer if done."""
    pending = state.get("pending_agents", []) or []
    executed = state.get("executed_agents", []) or []
    
    for agent in pending:
        if agent not in executed:
            if agent == "visual_search_agent":
                return "filter_extractor"
            return agent
            
    return "synthesizer"


AGENT_ROUTES = ["faq", "order", "filter_extractor", "synthesizer"]

# Register agent transitions using route_after_agent
builder.add_conditional_edges("faq", route_after_agent, AGENT_ROUTES)
builder.add_conditional_edges("order", route_after_agent, AGENT_ROUTES)

# Visual Search Pipeline
builder.add_edge("filter_extractor", "clip_embedder")
builder.add_edge("clip_embedder", "visual_search")
builder.add_edge("visual_search", "cleanup")
builder.add_conditional_edges("cleanup", route_after_agent, AGENT_ROUTES)

builder.add_edge("synthesizer", "summarizer")
builder.add_edge("summarizer", END)

def compile_graph(checkpointer=None):
    """Compiles and returns the graph, optionally attaching a checkpointer."""
    return builder.compile(checkpointer=checkpointer)

