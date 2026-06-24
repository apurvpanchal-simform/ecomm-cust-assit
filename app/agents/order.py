"""
Agent node for handling order lookups, item searches, and order tracking.

`order_node` is the Order specialist in the LangGraph multi-agent architecture.
It uses an agentic tool-calling loop (`generate_order_response`) where the LLM
can iteratively call Supabase DB tools to gather the data needed to answer the
user's question before producing a final text response.

Available tools:
    get_customer_orders  — list/filter the customer's orders
    get_order_details    — fetch complete details for a single order
    search_order_items   — keyword-search across all order line-items
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langfuse import observe

from app.config import AGENT_CONFIG, ORDER_LLM_CONFIG
from app.graph.state import AgentState
from app.prompts import get_order_system_prompt, get_order_chat_summary_message, get_order_sub_query_message
from app.services.llm_factory import get_llm
from app.tools.order_details import get_order_details
from app.tools.order_items import search_order_items
from app.tools.order_lookup import get_customer_orders

load_dotenv()

logger = logging.getLogger(__name__)


# ── Order Tools Config ────────────────────────────────────────────────────────
_ORDER_TOOLS = [get_customer_orders, get_order_details, search_order_items]
# Maps tool name strings → callable tool objects for dispatch in the loop
_TOOL_MAP: dict[str, Any] = {t.name: t for t in _ORDER_TOOLS}


# ── Private helpers ───────────────────────────────────────────────────────────


def _build_order_conversation(state: AgentState) -> list:
    """
    Assemble the initial conversation list for the order agent's tool-calling loop.

    Message order:
    1. System prompt — order agent identity, tool descriptions, rules, and customer ID context.
    2. Chat summary  — condensed older conversation (if present).
    3. Sub-query     — focused instruction from the supervisor (if present).
    4. Recent messages — full unsummarised conversation history (read-only context).

    Args:
        state: The global AgentState.

    Returns:
        A list of LangChain message objects ready for the first LLM call.
    """
    summarized_count = state.get("summarized_message_count", 0)
    recent_messages = state.get("messages", [])[summarized_count:]
    customer_id = state.get("customer_id", "")

    conversation = [SystemMessage(content=get_order_system_prompt(customer_id))]

    chat_summary = state.get("chat_summary", "")
    if chat_summary:
        conversation.append(get_order_chat_summary_message(chat_summary))

    sub_queries = state.get("sub_queries") or {}
    sub_query = sub_queries.get("order")
    if sub_query:
        conversation.append(get_order_sub_query_message(sub_query))

    conversation += recent_messages
    return conversation


def _extract_final_text(new_messages: list) -> str:
    """
    Find the last non-tool-call AIMessage in the message list and extract its text.

    The tool-calling loop may produce multiple AIMessages (one per tool invocation).
    We want only the final response message that contains the user-facing answer.

    Handles both plain string content and Anthropic-style list-of-blocks content.

    Args:
        new_messages: All messages generated during the tool-calling loop.

    Returns:
        The text content of the final AI response, or "No response generated."
    """
    # Walk backwards to find the last AIMessage that is NOT a tool-call dispatcher
    final_ai = next(
        (
            m
            for m in reversed(new_messages)
            if isinstance(m, AIMessage) and not m.tool_calls
        ),
        None,
    )

    if not final_ai:
        return "No response generated."

    # Handle Anthropic multimodal content format (list of typed blocks)
    if isinstance(final_ai.content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in final_ai.content
        )

    return str(final_ai.content)


# ── Agentic tool-calling loop ─────────────────────────────────────────────────


@observe(name="order_generation")
async def generate_order_response(
    agent,
    conversation: list,
    config: RunnableConfig,
    customer_id: str,
) -> tuple[str, list, str | None]:
    """
    Run the LLM → Tool → LLM agentic loop for order queries.

    Iteratively calls the LLM, processes any tool calls it emits, appends the
    tool results back to the conversation, and repeats until the LLM produces a
    final text response (no more tool calls) or the iteration limit is reached.

    Args:
        agent: The LLM instance with tools bound via `.bind_tools()`.
        conversation: The initial conversation list (system prompt + history).
        config: LangChain execution configuration.
        customer_id: The authenticated customer ID, injected into every tool call
                     so tools never rely on potentially-spoofed user input.

    Returns:
        A tuple of:
            resolution_text (str)      — the final user-facing answer
            new_messages    (list)     — all AI and Tool messages generated this turn
            error           (str|None) — "max_iterations_exceeded" or None
    """
    new_messages = []
    max_iters = AGENT_CONFIG.max_iterations

    for _ in range(max_iters):
        # ── Ask the LLM for the next action ───────────────────────────────────
        response = await agent.ainvoke(conversation, config=config)
        conversation.append(response)
        new_messages.append(response)

        # ── If no tool calls → LLM is done, break out of the loop ─────────────
        if not response.tool_calls:
            break

        # ── Execute all tool calls the LLM requested ──────────────────────────
        tool_msgs = []
        for tc in response.tool_calls:
            # Always inject customer_id so tools query the right customer's data
            args = {**tc.get("args", {}), "customer_id": customer_id}
            try:
                result = (
                    str(await _TOOL_MAP[tc["name"]].ainvoke(args)).strip()
                    or "No result returned."
                )
            except Exception as e:
                result = f"Error: {e}"

            tool_msgs.append(
                ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"])
            )

        # Append tool results so the LLM can reason over them on the next iteration
        conversation.extend(tool_msgs)
        new_messages.extend(tool_msgs)

    else:
        # Loop exhausted — return a safe fallback message
        msg = "I'm having trouble processing your order request. Please try again."
        new_messages.append(AIMessage(content=msg))
        return msg, new_messages, "max_iterations_exceeded"

    resolution_text = _extract_final_text(new_messages)
    return resolution_text, new_messages, None


# ── Node ──────────────────────────────────────────────────────────────────────


@observe(name="order_node")
async def order_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Handle user queries related to their personal order history and tracking.

    Flow:
    1. Validate that `customer_id` is present (injected by auth middleware).
       If missing, return an error message asking the user to sign in.
    2. Build the initial conversation (system prompt + history + sub-query).
    3. Bind the three order tools to the LLM.
    4. Run `generate_order_response` — the agentic tool-calling loop.
    5. Return all new messages (tool results + final AI answer) plus metadata.

    Note: The final AIMessage intentionally has no `name` field — Groq's API
    returns a 400 error if an AIMessage with a `name` field is included in
    conversation history on subsequent turns.

    Args:
        state: The global AgentState.
        config: LangChain execution configuration.

    Returns:
        A partial state dict with `messages`, `error`, and `executed_agents`.
    """
    customer_id = state.get("customer_id")

    # ── 1. Guard: customer must be authenticated ───────────────────────────────
    if not customer_id:
        msg = "Unable to verify your identity. Please sign in and try again."
        return {
            "messages": [AIMessage(content=msg)],
            "error": "missing_customer_id",
            "executed_agents": state.get("executed_agents", []) + ["order"],
        }

    # ── 2. Build the conversation ─────────────────────────────────────────────
    conversation = _build_order_conversation(state)

    # ── 3. Bind tools to the LLM ─────────────────────────────────────────────
    # cache=False is intentional — order data is live and user-specific;
    # caching would return stale results for different customers.
    llm = get_llm(
        temperature=ORDER_LLM_CONFIG.temperature,
        cache=ORDER_LLM_CONFIG.default_cache,
    )
    agent = llm.bind_tools(_ORDER_TOOLS)

    # ── 4. Run the agentic tool-calling loop ──────────────────────────────────
    try:
        resolution_text, new_messages, error = await generate_order_response(
            agent, conversation, config, customer_id
        )

        # Note: we intentionally do NOT set name="order" on the final AIMessage.
        # Groq's API returns a 400 error if an AIMessage with a 'name' field is
        # passed back into conversation history on subsequent turns.

    except Exception as exc:
        logger.error("order_node failed: %s", exc, exc_info=True)
        resolution_text = "I encountered an error while trying to process your order. Please try again."
        new_messages = [AIMessage(content=resolution_text)]
        error = str(exc)

    return {
        "messages": new_messages,
        "error": error,
        "executed_agents": state.get("executed_agents", []) + ["order"],
    }
