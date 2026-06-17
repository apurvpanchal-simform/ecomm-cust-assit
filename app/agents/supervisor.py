"""
The Supervisor agent node that routes user queries to the appropriate downstream agents.
"""

import logging

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import observe

from app.graph.state import AgentState
from app.graph.utils import filter_tool_messages
from app.schemas.agent import Route
from app.services.llm_factory import get_llm

load_dotenv()

SUPERVISOR_SYSTEM_PROMPT = """You are a routing supervisor for an e-commerce support system. You analyze the user's message and delegate work to the right agents.

## Agents
- `faq` → General questions: policies, shipping, returns, company info.
- `order` → Personal order data: tracking, refunds, order history, specific items purchased.
- `image_search_agent` → Product discovery: finding items by text description, uploaded image, or both.

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

## Search Filters (when routing to `image_search_agent`)
9. Extract a clean `search_query` combining user intent and image description. Exclude price terms from the query.
10. Set `min_price` / `max_price` if the user mentions a budget (e.g., "under $50" → max_price=50).

## Synthesis
A downstream synthesizer will merge all agent responses with your `response` field into one reply for the user. Write naturally.
"""


@observe(name="supervisor_node")
async def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Analyzes the conversation state and delegates work to specific agents.

    This node reads the user's latest messages, chat history summary, and any
    extracted image metadata. It then invokes an LLM using Structured Outputs
    to generate a routing plan (`Route`), which includes a list of pending agents,
    sub-queries for those agents, and search filters.

    Args:
        state: The global AgentState.
        config: The RunnableConfig containing thread execution context.

    Returns:
        A dictionary representing updates to the AgentState, primarily setting
        `pending_agents` and `sub_queries`.
    """

    if state.get("escalate_to_human"):
        return {
            "pending_agents": [],
            "executed_agents": [],
            "sub_queries": {},
            "error": None,
            "next": "FINISH",
            "messages": [
                AIMessage(
                    content="I am escalating this conversation to a human support agent. A representative will be with you shortly.",
                    name="supervisor",
                )
            ],
            "faq_chunks": [],
            "image_results": [],
        }

    messages = state.get("messages", [])

    llm = get_llm(temperature=0.0, cache=False if state.get("image_base64") or state.get("image_is_safe") is False else None)
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
        
        # If there's no safety warning, explicitly instruct the LLM that this is a safe image
        # and MUST be routed to the search agent, to prevent it from copying the blocked behavior
        # from the previous turn in the conversation history.
        if not state.get("image_safety_warning"):
            analysis_text.append("\nIMPORTANT: This image is SAFE and approved. You MUST route to 'image_search_agent'. DO NOT apologize or claim it was blocked due to policy.")

        supervisor_messages.append(
            SystemMessage(
                content="The user uploaded an image. Image Analysis:\n"
                + "\n".join(analysis_text)
            )
        )

    # Ephemeral safety warning from image_analyzer (only set for THIS turn)
    image_safety_warning = state.get("image_safety_warning")
    if image_safety_warning:
        supervisor_messages.append(
            SystemMessage(content=f"SYSTEM: {image_safety_warning}")
        )

    summarized_count = state.get("summarized_message_count", 0)
    recent_messages = filter_tool_messages(messages[summarized_count:])

    customer_id = state.get("customer_id", "guest")
    supervisor_messages.append(SystemMessage(content=f"Current Customer ID: {customer_id}"))

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
            p.strip() for p in pending if p in ["faq", "order", "image_search_agent"]
        ]
    except Exception as e:
        logging.getLogger(__name__).exception(f"Supervisor LLM Error: {e}")
        pending_str = []

    response_text = ""
    if response is not None:
        response_text = getattr(response, "response", "").strip()

    # Use explicit intent flag for images
    if state.get("image_base64") and getattr(response, "image_intent", False):
        if "image_search_agent" not in pending_str:
            pending_str.insert(0, "image_search_agent")

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
            "faq_chunks": [],
            "image_results": [],
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
        "faq_chunks": [],
        "image_results": [],
    }

    if response_text and pending_str:
        return_dict["messages"] = [AIMessage(content=response_text, name="supervisor")]

    return return_dict
