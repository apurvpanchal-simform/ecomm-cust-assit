from langgraph.graph import StateGraph, START, END
from app.agents.faq import faq_node
from app.agents.order import order_node
from app.agents.supervisor import supervisor_node
from app.graph.state import AgentState


def route_supervisor(state: AgentState) -> str:
    """Returns the next node to execute from the state."""
    next_node = state.get("next", "FINISH")
    if next_node == "FINISH":
        return END
    return next_node

builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("faq", faq_node)
builder.add_node("order", order_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "faq": "faq",
        "order": "order",
        END: END,
    }
)

builder.add_edge("faq", "supervisor")
builder.add_edge("order", "supervisor")

def compile_graph(checkpointer=None):
    """Compiles and returns the graph, optionally attaching a checkpointer."""
    return builder.compile(checkpointer=checkpointer)