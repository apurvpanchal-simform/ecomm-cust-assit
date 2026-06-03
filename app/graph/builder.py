from langgraph.graph import StateGraph, START, END
from app.agents.faq import faq_node
from app.agents.order import order_node
from app.graph.state import AgentState

builder = StateGraph(AgentState)

# builder.add_node("faq", faq_node)
builder.add_node("order", order_node)

builder.add_edge(START, "order")
builder.add_edge("order", END)

graph = builder.compile()