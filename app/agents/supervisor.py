"""
The Supervisor agent node that routes user queries to the appropriate downstream agents.
"""

import logging

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import observe

from app.config.llm_config import SUPERVISOR_LLM_CONFIG
from app.graph.state import AgentState
from app.graph.utils import filter_tool_messages
from app.prompts import get_supervisor_system_prompt
from app.schemas.agent import Route
from app.services.llm_factory import get_llm

load_dotenv()

logger = logging.getLogger(__name__)

# Valid downstream agent names the supervisor can route to
_VALID_AGENTS = {"faq", "order", "image_search_agent"}


# ── Private helpers ───────────────────────────────────────────────────────────


def _handle_escalation() -> dict:
    """Return the state update used when the conversation must be escalated to a human agent."""
    return {
        "pending_agents": [],
        "executed_agents": [],
        "sub_queries": {},
        "error": None,
        "next": "FINISH",
        "escalate_to_human": True,  # Explicitly persist so checkpoint stays consistent
        "messages": [
            AIMessage(
                content="I am escalating this conversation to a human support agent. A representative will be with you shortly."
            )
        ],
        "faq_chunks": [],
        "image_results": [],
    }


def _append_chat_summary(supervisor_messages: list, state: AgentState) -> None:
    """Inject the rolling chat summary as background context (capped to avoid prompt bloat)."""
    chat_summary = state.get("chat_summary", "")
    if not chat_summary:
        return
    if len(chat_summary) > 800:
        chat_summary = "[Truncated earlier summary]...\n" + chat_summary[-800:]
    supervisor_messages.append(
        SystemMessage(
            content=f"Background Context (Older info, lower priority):\n{chat_summary}"
        )
    )


def _append_image_context(supervisor_messages: list, state: AgentState) -> None:
    """Inject image description, tags, and safety warnings into the supervisor prompt."""
    image_desc = state.get("image_description")
    image_tags = state.get("image_tags")

    if image_desc or image_tags:
        analysis_text = []
        if image_desc:
            analysis_text.append(f"Description/OCR: {image_desc}")
        if image_tags:
            analysis_text.append(f"Tags: {', '.join(image_tags)}")

        # Explicitly instruct the LLM to route safely when no warning is present,
        # to prevent it from copying a blocked-image response from a previous turn.
        if not state.get("image_safety_warning"):
            analysis_text.append(
                "\nIMPORTANT: This image is SAFE and approved. You MUST route to 'image_search_agent'. DO NOT apologize or claim it was blocked due to policy."
            )

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


def _append_recent_messages(
    supervisor_messages: list, messages: list, state: AgentState
) -> None:
    """Append recent (unsummarized) conversation history to the supervisor prompt."""
    summarized_count = state.get("summarized_message_count", 0)
    recent_messages = filter_tool_messages(messages[summarized_count:])

    if recent_messages:
        supervisor_messages.append(
            SystemMessage(content="Latest Conversation (Highest priority):")
        )
        supervisor_messages.extend(recent_messages)


def _build_supervisor_messages(state: AgentState, messages: list) -> list:
    """Assemble the full ordered message list to send to the routing LLM."""
    supervisor_messages = [SystemMessage(content=get_supervisor_system_prompt())]
    _append_chat_summary(supervisor_messages, state)
    _append_image_context(supervisor_messages, state)
    _append_recent_messages(supervisor_messages, messages, state)
    return supervisor_messages


async def _invoke_llm(
    structured_llm, supervisor_messages: list, config: RunnableConfig
) -> tuple[Route | None, list[str]]:
    """Call the structured routing LLM and return (response, validated pending_agents list)."""
    try:
        response = await structured_llm.ainvoke(supervisor_messages, config=config)
        pending = response.pending_agents or []
        pending_str = [p.strip() for p in pending if p in _VALID_AGENTS]
        return response, pending_str
    except Exception as e:
        logger.exception("Supervisor LLM Error: %s", e)
        return None, []


def _build_finish_response(response_text: str) -> dict:
    """Return the state update for a no-agent-needed turn (chitchat, refusals, greetings)."""
    if not response_text:
        response_text = "You're welcome! Is there anything else I can help you with?"
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


def _build_routing_response(
    response: Route, pending_str: list[str], response_text: str
) -> dict:
    """Return the state update for a turn that routes to one or more downstream agents."""
    next_step = pending_str[0]

    active_filters: dict = {}
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

    # Include the supervisor's optional preamble alongside the routed agents' responses
    if response_text:
        return_dict["messages"] = [AIMessage(content=response_text)]

    return return_dict


# ── Public node ───────────────────────────────────────────────────────────────


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
    # 1. Short-circuit if a previous turn already triggered human escalation
    if state.get("escalate_to_human"):
        return _handle_escalation()

    messages = state.get("messages", [])

    # 2. Build the LLM
    has_image_context = (
        bool(state.get("image_base64")) or state.get("image_is_safe") is False
    )
    llm = get_llm(
        temperature=SUPERVISOR_LLM_CONFIG.temperature,
        cache=False if has_image_context else SUPERVISOR_LLM_CONFIG.default_cache,
    )
    structured_llm = llm.with_structured_output(Route)

    # 3. Build the prompt message list
    supervisor_messages = _build_supervisor_messages(state, messages)

    # 4. Call the routing LLM
    response, pending_str = await _invoke_llm(
        structured_llm, supervisor_messages, config
    )

    response_text = getattr(response, "response", "").strip() if response else ""

    # 5. Honour explicit image_intent flag from the LLM response
    if state.get("image_base64") and getattr(response, "image_intent", False):
        if "image_search_agent" not in pending_str:
            pending_str.insert(0, "image_search_agent")

    # 6. Return appropriate response based on routing decision
    if not pending_str:
        return _build_finish_response(response_text)

    return _build_routing_response(response, pending_str, response_text)
