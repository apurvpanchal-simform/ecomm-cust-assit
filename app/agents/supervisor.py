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

SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor managing a conversation between a user and three specialized agents:
- 'faq': Handles general questions about the company, policies, or generic operations (e.g., return policies, shipping times, generic info).
- 'order': Handles specific, personalized questions about the user's orders, tracking, refunds, or specific items they bought.
- 'visual_search_agent': Handles requests to find, search for, or buy products based on a visual description, an uploaded image, or a text search for items (e.g., "Do you have any black jacket?", "Show me red shoes").

Your task is to analyze the conversation history and the user's latest query, and determine which agents need to act to fully answer the query.

NOTE: After the selected agents finish executing, a "synthesizer" node will automatically combine their answers with any notes you provide in your `response` field. Therefore, write your refusal/clarification notes naturally, knowing they will be seamlessly combined with the agents' final answers to form a single, unified response for the user.

RULES:
1. Identify all parts of the user's query that require an agent to answer.
2. If a part requires general policy/info, include 'faq'.
3. If a part requires specific order lookup/refund/status, include 'order'.
4. If a part involves finding products, asking if a product is in stock, or acting on an uploaded image to find things, include 'visual_search_agent'.
5. MIXED INTENT RULE: If the user's message contains BOTH valid requests (e.g., order lookup) AND out-of-domain text (e.g., math, coding) OR a system note about a blocked image: You MUST route the valid parts to the correct agents, AND use the `response` field to write a brief note about what you cannot do.
6. If the query is COMPLETELY unrelated to our store (math, politics, etc.) and contains NO valid requests, leave `pending_agents` empty and write a polite refusal in the `response` field.
7. If the query IS related to our store or shopping but is just too broad or vague (e.g., "give me all products"), leave `pending_agents` empty and enthusiastically ask them to clarify what they are looking for in the `response` field.
8. If the user's message is ONLY a greeting, parting, gratitude, or simple chitchat, return an empty list for `pending_agents` and write a brief, warm 1-2 sentence response in the `response` field.
9. Do NOT answer the user's actual question yourself (except for greetings/chitchat or clarifications).
10. SEQUENTIAL EXECUTION RULE: You MUST order the `pending_agents` list logically. If Agent B needs information that will be found by Agent A, put Agent A first! (e.g., if finding an order's item category is needed before searching the FAQ for its warranty, put 'order' before 'faq').
11. SUB-QUERIES RULE: You MUST provide a precise `sub_queries` entry for each agent you trigger. Tell the agent exactly what its job is for this turn. If it needs to wait for data from a previous agent, explicitly tell it to "Read the conversation history to find X, then do Y".
"""


class Route(BaseModel):
    pending_agents: list[Literal["faq", "order", "visual_search_agent"]] = Field(
        description="The ordered list of agents to execute. Put data-gathering agents first if others depend on them. Return an empty list if the query is ONLY a greeting, chitchat, vague search, or completely out-of-domain."
    )
    sub_queries: dict[str, str] = Field(
        default_factory=dict,
        description="A dictionary mapping each pending agent name to its specific instructions (e.g., {'order': 'Find the last order', 'faq': 'Find the warranty for the item the order agent finds'})."
    )
    response: str = Field(
        default="",
        description="If you cannot fulfill part or all of the query (due to safety, out-of-domain, or vagueness), or if it's a greeting, provide your direct response/refusal here."
    )
    image_intent: bool = Field(
        default=False,
        description="Set to true if the user uploaded an image and wants to search or identify it."
    )


@traceable(name="supervisor_node", metadata={"agent": "supervisor"})
async def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    """Delegates to the correct agent or finishes the conversation."""

    messages = state.get("messages", [])

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(Route)

    supervisor_messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)]

    chat_summary = state.get("chat_summary", "")
    if chat_summary:
        # Cap summary to avoid prompt bloat
        if len(chat_summary) > 800:
            chat_summary = "[Truncated earlier summary]...\n" + chat_summary[-800:]
        supervisor_messages.append(SystemMessage(content=f"Background Context (Older info, lower priority):\n{chat_summary}"))

    image_desc = state.get("image_description")
    image_tags = state.get("image_tags")
    if image_desc or image_tags:
        analysis_text = []
        if image_desc: analysis_text.append(f"Description/OCR: {image_desc}")
        if image_tags: analysis_text.append(f"Tags: {', '.join(image_tags)}")
        supervisor_messages.append(SystemMessage(content=f"The user uploaded an image. Image Analysis:\n" + "\n".join(analysis_text)))

    summarized_count = state.get("summarized_message_count", 0)
    recent_messages = messages[summarized_count:]
    if recent_messages:
        supervisor_messages.append(SystemMessage(content="Latest Conversation (Highest priority):"))
    for msg in recent_messages:
        if getattr(msg, "type", "") == "ai" and isinstance(msg.content, str) and len(msg.content) > 500:
            # Create a truncated copy to save tokens
            truncated_content = msg.content[:500] + "\n... [Truncated for brevity]"
            truncated_msg = AIMessage(
                content=truncated_content, 
                name=getattr(msg, "name", None),
                tool_calls=getattr(msg, "tool_calls", [])
            )
            supervisor_messages.append(truncated_msg)
        else:
            supervisor_messages.append(msg)

    response = None
    try:
        response = await structured_llm.ainvoke(supervisor_messages, config=config)
        pending = response.pending_agents or []
        pending_str = [p.strip() for p in pending if p in ["faq", "order", "visual_search_agent"]]
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception(f"Supervisor LLM Error: {e}")
        pending_str = []
        
    response_text = ""
    if response is not None:
        response_text = getattr(response, "response", "").strip()

    # Use explicit intent flag for images
    if state.get("image_base64") and getattr(response, "image_intent", False):
        if "visual_search_agent" not in pending_str:
            pending_str.insert(0, "visual_search_agent")

    # When the supervisor decides no pending agents, we return the response generated in the same call.
    if not pending_str:
        if not response_text:
            response_text = "You're welcome! Is there anything else I can help you with?"
            
        return {
            "pending_agents": [],
            "executed_agents": [],
            "sub_queries": {},
            "error": None,
            "next": "FINISH",
            "messages": [AIMessage(content=response_text)],
        }

    next_step = pending_str[0] if pending_str else "FINISH"
    return_dict = {
        "pending_agents": pending_str,
        "executed_agents": [],
        "sub_queries": getattr(response, "sub_queries", {}),
        "error": None,
        "next": next_step
    }
    
    if response_text and pending_str:
        return_dict["messages"] = [AIMessage(content=response_text, name="supervisor")]
        
    return return_dict
