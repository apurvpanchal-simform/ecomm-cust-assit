import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langsmith import traceable
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from app.services.llm import get_llm

from app.graph.state import AgentState

load_dotenv()

SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor managing a conversation between a user and two specialized agents:
- 'faq': Handles general questions about the company, policies, or generic operations (e.g., return policies, shipping times, generic info).
- 'order': Handles specific, personalized questions about the user's orders, tracking, refunds, or specific items they bought.
- 'visual_search_agent': Handles requests to find, search for, or buy products based on a visual description, an uploaded image, or a text search for similar items.

Your task is to analyze the conversation history and the user's latest query, and determine which agents need to act to fully answer the query.

RULES:
1. Identify all parts of the user's query that require an agent to answer.
2. If a part requires general policy/info, include 'faq'.
3. If a part requires specific order lookup/refund/status, include 'order'.
4. If a part involves finding products or acting on an uploaded image to find things, include 'visual_search_agent'.
5. If the query contains MULTIPLE, return them in the list `pending_agents`, ordering them logically.
5. If the query is NOT related to FAQs and NOT related to orders, include 'out_of_domain'.
6. If the user's message is a greeting, parting, gratitude, or simple chitchat, return an empty list for `pending_agents` and write a brief, warm 1-2 sentence response in the `response` field.
7. Do NOT answer the user's actual question yourself (except for greetings/chitchat).
"""


class Route(BaseModel):
    pending_agents: list[Literal["faq", "order", "visual_search_agent", "out_of_domain"]] = Field(
        description="The list of agents that need to be executed to answer all parts of the user's query. Return an empty list if resolved or if the query is a greeting/chitchat."
    )
    response: str = Field(
        default="",
        description="If pending_agents is empty and the user's message is a greeting, farewell, or gratitude (like 'hi', 'thanks', 'bye'), provide a warm, brief 1-2 sentence response. Otherwise leave empty."
    )


@traceable(name="supervisor_node", metadata={"agent": "supervisor"})
async def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    """Delegates to the correct agent or finishes the conversation."""

    messages = state.get("messages", [])

    # No hard-coded image routing anymore. Supervisor decides based on image_description.

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(Route)

    supervisor_messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)]

    chat_summary = state.get("chat_summary", "")
    if chat_summary:
        supervisor_messages.append(SystemMessage(content=f"Summary of earlier conversation:\n{chat_summary}"))

    image_desc = state.get("image_description")
    if image_desc:
        supervisor_messages.append(SystemMessage(content=f"The user uploaded an image. Image Analysis:\n{image_desc}"))

    summarized_count = state.get("summarized_message_count", 0)
    recent_messages = messages[summarized_count:]
    for msg in recent_messages:
        supervisor_messages.append(msg)

    response = None
    try:
        response = await structured_llm.ainvoke(supervisor_messages, config=config)
        pending = response.pending_agents or []
        pending_str = [p.strip() for p in pending if p in ["faq", "order", "visual_search_agent", "out_of_domain"]]
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception(f"Supervisor LLM Error: {e}")
        pending_str = []
        
    # Bypass for images: guarantee visual_search_agent runs for images, 
    # UNLESS the supervisor explicitly marked the image as out_of_domain.
    if state.get("image_base64"):
        if "out_of_domain" not in pending_str and "visual_search_agent" not in pending_str:
            pending_str.insert(0, "visual_search_agent")

    # When the supervisor decides no pending agents and the last message is from the user
    # (meaning chitchat or greeting), we return the response generated in the same call.
    if not pending_str and messages and hasattr(messages[-1], 'type') and messages[-1].type == 'human':
        response_text = ""
        if response is not None:
            response_text = getattr(response, "response", "").strip()
        if not response_text:
            response_text = "You're welcome! Is there anything else I can help you with?"
            
        return {
            "pending_agents": [],
            "executed_agents": [],
            "next": "FINISH",
            "messages": [AIMessage(content=response_text)],
        }

    next_step = pending_str[0] if pending_str else "FINISH"
    return {
        "pending_agents": pending_str,
        "executed_agents": [],
        "next": next_step
    }


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
async def out_of_domain_node(state: AgentState, config: RunnableConfig) -> dict:
    """Returns a friendly message when the user's query is outside the supported domain."""
    return {
        "messages": [AIMessage(content=OUT_OF_DOMAIN_MESSAGE)],
        "agent_response": {
            "resolution_text": OUT_OF_DOMAIN_MESSAGE,
        },
        "executed_agents": state.get("executed_agents", []) + ["out_of_domain"],
    }
