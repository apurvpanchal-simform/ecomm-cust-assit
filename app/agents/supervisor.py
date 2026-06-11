import os

from dotenv import load_dotenv
from langsmith import traceable
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from app.services.llm import get_llm

from app.graph.state import AgentState
from app.graph.utils import filter_tool_messages
from app.schemas.agent import Route

load_dotenv()

SUPERVISOR_SYSTEM_PROMPT = """You are a routing supervisor for an e-commerce support system. You analyze the user's message and delegate work to the right agents.

## Agents
- `faq` → General questions: policies, shipping, returns, company info.
- `order` → Personal order data: tracking, refunds, order history, specific items purchased.
- `visual_search_agent` → Product discovery: finding items by text description, uploaded image, or both.

## Routing Rules
1. Route each distinct part of the user's query to the correct agent(s).
2. For mixed-intent messages (e.g., valid request + out-of-domain text), route the valid parts AND write a brief note in `response` about what you cannot help with.
3. For completely off-topic queries (math, politics, coding), return empty `pending_agents` and a polite refusal in `response`.
4. For vague shopping queries ("show me everything"), return empty `pending_agents` and ask for clarification in `response`.
5. For greetings, thanks, or chitchat, return empty `pending_agents` and a warm 1-2 sentence `response`.
6. Never answer the user's actual question yourself—only route or respond to chitchat.

## Execution Rules
7. Order `pending_agents` by dependency: if Agent B needs Agent A's output, list A first.
8. Provide a precise `sub_queries` entry for every agent you trigger, telling it exactly what to do this turn.

## Search Filters (when routing to `visual_search_agent`)
9. Extract a clean `search_query` combining user intent and image description. Exclude price terms from the query.
10. Set `min_price` / `max_price` if the user mentions a budget (e.g., "under $50" → max_price=50).

## Synthesis
A downstream synthesizer will merge all agent responses with your `response` field into one reply for the user. Write naturally.
"""


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
        supervisor_messages.append(
            SystemMessage(
                content=f"Background Context (Older info, lower priority):\n{chat_summary}"
            )
        )

    image_desc = state.get("image_description")
    image_tags = state.get("image_tags")
    if image_desc or image_tags:
        analysis_text = []
        if image_desc:
            analysis_text.append(f"Description/OCR: {image_desc}")
        if image_tags:
            analysis_text.append(f"Tags: {', '.join(image_tags)}")
        supervisor_messages.append(
            SystemMessage(
                content=f"The user uploaded an image. Image Analysis:\n"
                + "\n".join(analysis_text)
            )
        )

    summarized_count = state.get("summarized_message_count", 0)
    recent_messages = filter_tool_messages(messages[summarized_count:])
    if recent_messages:
        supervisor_messages.append(
            SystemMessage(content="Latest Conversation (Highest priority):")
        )
    for msg in recent_messages:
        if (
            getattr(msg, "type", "") == "ai"
            and isinstance(msg.content, str)
            and len(msg.content) > 500
        ):
            # Create a truncated copy to save tokens
            truncated_content = msg.content[:500] + "\n... [Truncated for brevity]"
            truncated_msg = AIMessage(
                content=truncated_content,
                name=getattr(msg, "name", None),
                tool_calls=getattr(msg, "tool_calls", []),
            )
            supervisor_messages.append(truncated_msg)
        else:
            supervisor_messages.append(msg)

    response = None
    try:
        response = await structured_llm.ainvoke(supervisor_messages, config=config)
        pending = response.pending_agents or []
        pending_str = [
            p.strip() for p in pending if p in ["faq", "order", "visual_search_agent"]
        ]
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
            response_text = (
                "You're welcome! Is there anything else I can help you with?"
            )

        return {
            "pending_agents": [],
            "executed_agents": [],
            "sub_queries": {},
            "error": None,
            "next": "FINISH",
            "messages": [AIMessage(content=response_text)],
        }

    next_step = pending_str[0] if pending_str else "FINISH"

    active_filters = {}
    if getattr(response, "min_price", None) is not None:
        active_filters["min_price"] = response.min_price
    if getattr(response, "max_price", None) is not None:
        active_filters["max_price"] = response.max_price

    return_dict = {
        "pending_agents": pending_str,
        "executed_agents": [],
        "sub_queries": getattr(response, "sub_queries", {}),
        "search_query": getattr(response, "search_query", None),
        "active_filters": active_filters if active_filters else None,
        "error": None,
        "next": next_step,
    }

    if response_text and pending_str:
        return_dict["messages"] = [AIMessage(content=response_text, name="supervisor")]

    return return_dict
