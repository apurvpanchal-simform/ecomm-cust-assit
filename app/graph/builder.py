from langgraph.graph import StateGraph, START, END
from app.agents.faq import faq_node
from app.agents.order import order_node
from app.agents.router import router_node
from app.graph.state import AgentState


def route_intent(state: AgentState) -> str:
    """Returns the intent string from the state to route to the correct node."""
    return state.get("intent", "faq")

builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("faq", faq_node)
builder.add_node("order", order_node)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    route_intent,
    {
        "faq": "faq",
        "order": "order",
    }
)

builder.add_edge("faq", END)
builder.add_edge("order", END)

def compile_graph(checkpointer=None):
    """Compiles and returns the graph, optionally attaching a checkpointer."""
    return builder.compile(checkpointer=checkpointer)