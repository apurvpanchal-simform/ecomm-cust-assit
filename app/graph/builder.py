from langgraph.graph import StateGraph, START, END
from app.agents.faq import faq_node
from app.agents.order import order_node
from app.agents.supervisor import supervisor_node, out_of_domain_node
from app.agents.summarizer import summarizer_node
from app.graph.state import AgentState


def route_supervisor(state: AgentState) -> str:
    """Returns the next node to execute from the state."""
    next_node = state.get("next", "FINISH")
    if next_node == "FINISH":
        return "summarizer"
    return next_node


builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("faq", faq_node)
builder.add_node("order", order_node)
builder.add_node("out_of_domain", out_of_domain_node)
builder.add_node("summarizer", summarizer_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "faq": "faq",
        "order": "order",
        "out_of_domain": "out_of_domain",
        "summarizer": "summarizer",
    },
)

builder.add_edge("faq", "supervisor")
builder.add_edge("order", "supervisor")
builder.add_edge("out_of_domain", "summarizer")
builder.add_edge("summarizer", END)


def compile_graph(checkpointer=None):
    """Compiles and returns the graph, optionally attaching a checkpointer."""
    return builder.compile(checkpointer=checkpointer)

