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

ROUTER_SYSTEM_PROMPT = """You are a smart routing assistant for an e-commerce platform.
Your job is to read the user's query and classify it into exactly one of two intents: "FAQ" or "ORDER".

─── FAQ Intent ───
Route to "FAQ" if the user is asking a GENERAL question about the company, policies, or generic operations.
Examples: 
- "What is your return policy?"
- "How long does shipping usually take?"
- "Do you ship internationally?"
- "What payment methods do you accept?"

─── ORDER Intent ───
Route to "ORDER" if the user is asking a SPECIFIC, PERSONALIZED question about their own purchases, tracking, or items.
Examples: 
- "Where is my package?"
- "Can I return the shirt I bought yesterday?"
- "Did my refund process for order ORD-123?"
- "What did I order last month?"

Analyze the query and return the correct intent as a JSON object. If it is ambiguous but mentions a specific item, package, or order, default to ORDER.
"""

class RouterOutput(BaseModel):
    intent: str = Field(
        description="The classified intent of the user's query. Must be exactly 'faq' or 'order'."
    )

@traceable(name="router_node", metadata={"agent": "router"})
def router_node(state: AgentState, config: RunnableConfig) -> dict:
    """Classifies the user query and updates the state with the intent."""
    
    query = state.get("query", "")
    
    if not query:
        return {**state, "intent": "faq"}

    model_name = os.getenv("PRIMARY_MODEL")
    if not model_name:
        raise RuntimeError("PRIMARY_MODEL environment variable is not set.")
        
    llm = ChatGroq(model=model_name, temperature=0.0)
    structured_llm = llm.with_structured_output(RouterOutput, method="json_mode")

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    try:
        response = structured_llm.invoke(messages, config=config)
        intent = response.intent.lower().strip()
        if intent not in ["faq", "order"]:
            intent = "faq"
    except Exception as e:
        print(f"Router LLM Error: {e}")
        intent = "faq"

    return {**state, "intent": intent}
