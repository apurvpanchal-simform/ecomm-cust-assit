import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langsmith import traceable
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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
6. If the user's query is NOT related to FAQs (company policies, shipping, returns, general info) and NOT related to orders (tracking, refunds, order details, items purchased), output 'out_of_domain'.
7. You MUST return your answer as a JSON object with a single key "next", whose value is one of "faq", "order", "out_of_domain", or "FINISH".
"""


class Route(BaseModel):
    next: Literal["faq", "order", "visual_search_agent", "out_of_domain", "FINISH"] = Field(
        description="The next agent to call, 'out_of_domain' if the query is unrelated, or 'FINISH' if resolved."
    )


@traceable(name="supervisor_node", metadata={"agent": "supervisor"})
def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    """Delegates to the correct agent or finishes the conversation."""

    query = state.get("query", "")
    messages = state.get("messages", [])

    # Short-circuit: if an image is present, route to visual search pipeline
    if state.get("image_base64"):
        return {"next": "visual_search_agent"}


    model_name = os.getenv("PRIMARY_MODEL")
    if not model_name:
        raise RuntimeError("PRIMARY_MODEL environment variable is not set.")

    llm = ChatGroq(model=model_name, temperature=0.0)
    structured_llm = llm.with_structured_output(Route, method="json_mode")

    supervisor_messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)]

    chat_summary = state.get("chat_summary", "")
    if chat_summary:
        supervisor_messages.append(SystemMessage(content=f"Summary of earlier conversation:\n{chat_summary}"))

    # Only send the last 4 messages to the supervisor for routing decisions.
    # The query is already in messages as a HumanMessage (added by the /chat endpoint).
    recent_messages = messages[-4:] if len(messages) > 4 else messages
    for msg in recent_messages:
        supervisor_messages.append(msg)

    try:
        response = structured_llm.invoke(supervisor_messages, config=config)
        next_step = response.next.strip()
        if next_step not in ["faq", "order", "visual_search_agent", "out_of_domain", "FINISH"]:
            next_step = "FINISH"
    except Exception as e:
        print(f"Supervisor LLM Error: {e}")
        next_step = "FINISH"

    # When the supervisor decides FINISH and the last message is from the user
    # (meaning no agent ran on this turn), we need to generate a brief AI response.
    # Otherwise extract_response_text will pick up the stale response from the previous turn.
    if next_step == "FINISH" and messages and hasattr(messages[-1], 'type') and messages[-1].type == 'human':
        # Generate a brief conversational response
        brief_llm = ChatGroq(model=model_name, temperature=0.3)
        brief_messages = [
            SystemMessage(content="You are a friendly customer support assistant. The user has sent a conversational message (like 'thank you', 'okay', etc.). Respond briefly and warmly. Ask if they need anything else. Keep it to 1-2 sentences."),
        ]
        if chat_summary:
            brief_messages.append(SystemMessage(content=f"Context from earlier conversation:\n{chat_summary}"))
        brief_messages.append(messages[-1])
        
        try:
            brief_response = brief_llm.invoke(brief_messages, config=config)
            return {
                "next": next_step,
                "messages": [AIMessage(content=brief_response.content)],
            }
        except Exception:
            return {
                "next": next_step,
                "messages": [AIMessage(content="You're welcome! Is there anything else I can help you with?")],
            }

    return {"next": next_step}


OUT_OF_DOMAIN_MESSAGE = (
    "I'm sorry, but your question falls outside the areas I can help with. "
    "I specialize in order support and general company FAQs.\n\n"
    "Here are some examples of questions I can answer:\n"
    "• \"Where is my order?\"\n"
    "• \"What is your return policy?\"\n"
    "• \"Did I ever order a laptop?\"\n"
    "• \"Show me my recent orders\"\n"
    "• \"How long does shipping take?\"\n\n"
    "Please try rephrasing your question within these topics!"
)


@traceable(name="out_of_domain_node", metadata={"agent": "out_of_domain"})
def out_of_domain_node(state: AgentState, config: RunnableConfig) -> dict:
    """Returns a friendly message when the user's query is outside the supported domain."""
    return {
        "messages": [AIMessage(content=OUT_OF_DOMAIN_MESSAGE)],
        "agent_response": {
            "resolution_text": OUT_OF_DOMAIN_MESSAGE,
            "confidence_score": 1.0,
            "ticket_category": "out_of_domain",
            "requires_human": False,
            "escalation_reason": None,
        },
    }
