import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langsmith import traceable
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.graph.state import AgentState

load_dotenv()

SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor managing a conversation between a user and two specialized agents:
- 'faq': Handles general questions about the company, policies, or generic operations (e.g., return policies, shipping times, generic info).
- 'order': Handles specific, personalized questions about the user's orders, tracking, refunds, or specific items they bought.

Your task is to analyze the conversation history and the user's latest query, and determine who should act next.

RULES:
1. If the user just asked a question and it hasn't been answered, route to 'faq' or 'order'.
2. If an agent has provided an answer that satisfies the user's request, output 'FINISH'.
3. If the user's request is ambiguous but mentions a specific item, package, or order, default to 'order'.
4. MULTI-PART QUESTIONS: If the user's query contains BOTH an FAQ question and an Order question, route to one agent first (e.g., 'order'). Once that agent answers, the conversation will return to you. You MUST then route to the other agent (e.g., 'faq') to answer the remaining part. Do NOT output 'FINISH' until all parts of the user's query have been fully addressed.
5. Do NOT answer the user's question yourself. Your only job is to route to the correct agent or 'FINISH'.
6. You MUST return your answer as a JSON object with a single key "next", whose value is one of "faq", "order", or "FINISH".
"""

class Route(BaseModel):
    next: Literal["faq", "order", "FINISH"] = Field(
        description="The next agent to call, or 'FINISH' if the user's request is resolved."
    )

@traceable(name="supervisor_node", metadata={"agent": "supervisor"})
def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    """Delegates to the correct agent or finishes the conversation."""
    
    query = state.get("query", "")
    messages = state.get("messages", [])
    
    model_name = os.getenv("PRIMARY_MODEL")
    if not model_name:
        raise RuntimeError("PRIMARY_MODEL environment variable is not set.")
        
    llm = ChatGroq(model=model_name, temperature=0.0)
    structured_llm = llm.with_structured_output(Route, method="json_mode")

    supervisor_messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)]
    
    if query:
        supervisor_messages.append(HumanMessage(content=f"Initial User Query: {query}"))
        
    for msg in messages:
        supervisor_messages.append(msg)

    try:
        response = structured_llm.invoke(supervisor_messages, config=config)
        next_step = response.next.strip()
        if next_step not in ["faq", "order", "FINISH"]:
            next_step = "FINISH"
    except Exception as e:
        print(f"Supervisor LLM Error: {e}")
        next_step = "FINISH"

    return {"next": next_step}
