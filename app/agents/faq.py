"""
Agent node that handles general FAQ, policy, and shipping inquiries using RAG.

`faq_node` is the FAQ specialist in the LangGraph multi-agent architecture.
When the supervisor routes a query here, the node:
1. Checks the semantic response cache (Redis VSS) — returns immediately on a hit.
2. Directly invokes the `search_faq` tool to fetch relevant knowledge chunks.
3. Injects the retrieved context into the LLM prompt to prevent hallucination.
4. Calls the LLM to generate a grounded answer.
5. Writes the answer back to the semantic cache for future similar queries.
"""

from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import observe

from app.config.llm_config import FAQ_LLM_CONFIG
from app.graph.state import AgentState
from app.graph.utils import filter_tool_messages
from app.prompts import get_faq_system_prompt, get_faq_chat_summary_message, get_faq_sub_query_message
from app.services.faq_response_cache import get_faq_response, set_faq_response
from app.services.llm_factory import get_llm
from app.tools.faq_search import search_faq

load_dotenv()

_FAQ_TOOLS = [search_faq]
_TOOL_MAP: dict[str, Any] = {t.name: t for t in _FAQ_TOOLS}


# ── Private helpers ───────────────────────────────────────────────────────────


def _get_search_term(state: AgentState) -> str:
    """
    Determine the best search term to use for the FAQ vector search.

    Prefers the supervisor's focused `sub_query` for 'faq' if available,
    otherwise falls back to the raw user query.

    Args:
        state: The global AgentState.

    Returns:
        The search string to pass to `search_faq`.
    """
    sub_queries = state.get("sub_queries") or {}
    sub_query = sub_queries.get("faq")
    return sub_query if sub_query else state.get("query", "")


async def _fetch_faq_context(search_term: str) -> tuple[str, list]:
    """
    Invoke the `search_faq` tool and return the context string and raw chunks.

    Handles errors gracefully — returns an error message as context rather than
    crashing the node, so the LLM can at least apologise to the user.

    Args:
        search_term: The query string to search the FAQ knowledge base with.

    Returns:
        A tuple of (context_string, faq_chunks_list).
    """
    try:
        search_result = await search_faq.ainvoke({"query": search_term})
        if isinstance(search_result, dict):
            context = str(search_result.get("context", "")).strip()
            faq_chunks = search_result.get("chunks", [])
        else:
            context = str(search_result).strip()
            faq_chunks = []
    except Exception as e:
        context = f"Error performing search: {e}"
        faq_chunks = []

    return context, faq_chunks


def _build_conversation(
    state: AgentState,
    context: str,
    recent_messages: list,
) -> list:
    """
    Assemble the ordered list of messages to pass to the FAQ LLM.

    Message order (highest to lowest priority for the LLM):
    1. System prompt — FAQ agent identity, rules, and retrieved knowledge context.
    2. Chat summary   — condensed older conversation context (if present).
    3. Sub-query task — focused instruction from the supervisor (if present).
    4. Recent messages — latest conversation turns (highest priority context).

    Args:
        state: The global AgentState.
        context: The aggregated FAQ knowledge base context string.
        recent_messages: Unsummarised messages from this turn onwards.

    Returns:
        A list of LangChain message objects ready for LLM invocation.
    """
    sub_queries = state.get("sub_queries") or {}
    sub_query = sub_queries.get("faq")

    conversation = [SystemMessage(content=get_faq_system_prompt(context))]

    # Include rolling summary of older turns as background context
    chat_summary = state.get("chat_summary", "")
    if chat_summary:
        conversation.append(get_faq_chat_summary_message(chat_summary))

    # Give the agent a focused task for this specific turn (set by supervisor)
    if sub_query:
        conversation.append(get_faq_sub_query_message(sub_query))

    # Append recent conversation history (tool messages stripped to avoid provider errors)
    conversation += filter_tool_messages(recent_messages)

    return conversation


# ── Node ──────────────────────────────────────────────────────────────────────


@observe(name="faq_node")
async def faq_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Generate a grounded FAQ answer using RAG (Retrieve-then-Generate).

    Flow:
    1. Extract the search term (sub_query or raw query).
    2. Check the semantic response cache — return immediately on a hit.
    3. Fetch FAQ context chunks from Qdrant via `search_faq`.
    4. Build the LLM conversation (system prompt + summary + context + history).
    5. Invoke the LLM to generate a context-grounded answer.
    6. Cache the response if real chunks were retrieved (prevents caching apologies).

    Args:
        state: The global AgentState.
        config: LangChain execution configuration.

    Returns:
        A partial state dict with:
            messages        — list containing the generated AIMessage
            error           — error string or None
            executed_agents — updated list marking 'faq' as done
            faq_chunks      — raw Qdrant chunks used for the answer
    """
    # ── 1. Determine search term and query for cache lookup ───────────────────
    summarized_count = state.get("summarized_message_count", 0)
    recent_messages = state.get("messages", [])[summarized_count:]

    search_term = _get_search_term(state)
    customer_id = state.get("customer_id", "guest")
    raw_query = state.get("query", "")

    # ── 2. Check semantic response cache ──────────────────────────────────────
    # Skip cache when image context is present — image-aware responses must not
    # be served from cache since they depend on the specific image.
    has_image_context = (
        bool(state.get("image_base64")) or state.get("image_is_safe") is False
    )
    cached_response = (
        await get_faq_response(customer_id, raw_query)
        if not has_image_context
        else None
    )
    if cached_response:
        return {
            "messages": [AIMessage(content=cached_response)],
            "error": None,
            "executed_agents": state.get("executed_agents", []) + ["faq"],
            "faq_chunks": [],
        }

    # ── 3. Retrieve FAQ context from Qdrant ───────────────────────────────────
    context, faq_chunks = await _fetch_faq_context(search_term)

    # ── 4. Build the prompt conversation ──────────────────────────────────────
    conversation = _build_conversation(
        state=state,
        context=context,
        recent_messages=recent_messages,
    )

    # ── 5. Invoke the LLM ─────────────────────────────────────────────────────
    llm = get_llm(
        temperature=FAQ_LLM_CONFIG.temperature,
        cache=False if has_image_context else FAQ_LLM_CONFIG.default_cache,
    )

    try:
        response = await llm.ainvoke(conversation, config=config)
        resolution_text = str(response.content).strip()
        new_messages = [AIMessage(content=resolution_text)]
        error = None

        # ── 6. Write to cache (only when real chunks were found) ──────────────
        # faq_chunks=[] means Qdrant was unavailable or returned nothing, so the
        # LLM likely generated an apologetic fallback — don't cache that.
        if resolution_text and faq_chunks and not has_image_context:
            await set_faq_response(customer_id, raw_query, resolution_text)

    except Exception as exc:
        resolution_text = "I encountered an error while trying to process your request. Please try again."
        new_messages = [AIMessage(content=resolution_text)]
        error = str(exc)

    return {
        "messages": new_messages,
        "error": error,
        "executed_agents": state.get("executed_agents", []) + ["faq"],
        "faq_chunks": faq_chunks,
    }
