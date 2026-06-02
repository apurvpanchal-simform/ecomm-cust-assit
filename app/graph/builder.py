from langgraph.graph import StateGraph, START, END
from app.agents.faq import faq_node
from app.graph.state import AgentState

builder = StateGraph(AgentState)

builder.add_node("faq", faq_node)

builder.add_edge(START, "faq")
builder.add_edge("faq", END)

graph = builder.compile()