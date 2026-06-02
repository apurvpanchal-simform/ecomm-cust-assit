from langgraph.graph import StateGraph, START, END
from app.agents.faq import faq_node
from app.agents.order import order_node
from app.graph.state import AgentState

builder = StateGraph(AgentState)

def route_selector(state: AgentState):
    return state.get("route")

builder.add_node("order", order_node)

builder.add_edge(START, "order")
builder.add_edge("order", END)

graph = builder.compile()