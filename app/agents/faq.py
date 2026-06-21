"""
Agent node that handles general FAQ, policy, and shipping inquiries using RAG.
"""

from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import observe

from app.graph.state import AgentState
from app.graph.utils import filter_tool_messages
from app.services.faq_response_cache import get_faq_response, set_faq_response
from app.services.llm_factory import get_llm
from app.tools.faq_search import search_faq

load_dotenv()

_FAQ_TOOLS = [search_faq]
_TOOL_MAP: dict[str, Any] = {t.name: t for t in _FAQ_TOOLS}

SYSTEM_PROMPT = """You are a FAQ support agent for an e-commerce platform. You answer questions about policies, shipping, returns, and general company info.

## Rules
1. Always use the `search_faq` tool to retrieve context before answering.
2. Answer strictly from the retrieved context—never invent policies or facts.
3. If the answer isn't in the context, say so and suggest contacting human support.
4. If the context or data you need is already in the conversation summary, use it directly without a duplicate tool call.
5. If `search_faq` returns an error, do NOT retry. Apologize and explain the service is temporarily unavailable.
6. If you receive a "specific task for this turn", prioritize that task over unrelated conversation.
7. Be professional, concise, and friendly.
8. If the retrieved context contains any exceptions or special conditions (e.g., non-returnable items, warranty exclusions), you MUST explicitly state them.
"""


@observe(name="faq_node")
async def faq_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Handles user queries related to general platform questions, policies, and FAQs.

    This node is part of the LangGraph multi-agent architecture and acts as the FAQ specialist.
    When the supervisor delegates a query to this node, it automatically searches the FAQ
    knowledge base using the provided sub-query or the original user query, and then formulates
    a response using only the retrieved context.

    Flow:
    1. Extracts the relevant query (`sub_query` or `query`) from the state.
    2. Directly invokes the `search_faq` tool to retrieve relevant documentation.
    3. Injects the retrieved context directly into the system prompt to prevent hallucination.
    4. Constructs a conversation array consisting of the system prompt, chat summary,
       task instructions, retrieved context, and recent conversation history.
    5. Invokes the LLM to generate an answer based purely on the context.
    6. Appends the AI response to the state and marks the agent as executed.

    Args:
        state (AgentState): The global state of the conversation, containing history,
                            sub-queries, and execution tracking.
        config (RunnableConfig): Configuration parameters for LangChain execution.

    Returns:
        dict: A dictionary containing:
            - `messages`: A list containing the newly generated AIMessage.
            - `error`: An error string if an exception occurred, otherwise None.
            - `executed_agents`: The updated list of agents that have run in this turn.
    """

    # In Sequential List without isolation, we just read all recent messages
    summarized_count = state.get("summarized_message_count", 0)
    recent_messages = state.get("messages", [])[summarized_count:]

    sub_queries = state.get("sub_queries") or {}
    sub_query = sub_queries.get("faq")
    search_term = sub_query if sub_query else state.get("query", "")

    customer_id = state.get("customer_id", "guest")
    raw_query = state.get("query", "")

    # --- Application-level Response Cache (5 min TTL) ---
    # Keyed by (customer_id, normalized_query). Works across same-chat repeated questions.
    # Do not cache or check cache if there is an image, OR if an image was blocked for safety.
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

    # Pre-fetch context using the search tool directly in Python
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

    context_message = SystemMessage(
        content=(
            f"Here is the context retrieved from the FAQ knowledge base for the user's query:\n"
            f"[[CONTEXT START]]\n"
            f"{context}\n"
            f"[[CONTEXT END]]\n\n"
            f"Answer the user's query using ONLY the provided context. Follow all guidelines."
        )
    )

    conversation = [SystemMessage(content=SYSTEM_PROMPT)]

    conversation.append(SystemMessage(content=f"Current Customer ID: {customer_id}"))

    chat_summary = state.get("chat_summary", "")
    if chat_summary:
        conversation.append(
            SystemMessage(content=f"Summary of earlier conversation:\n{chat_summary}")
        )

    if sub_query:
        conversation.append(
            SystemMessage(content=f"Your specific task for this turn: {sub_query}")
        )

    conversation.append(context_message)
    conversation += filter_tool_messages(recent_messages)

    llm = get_llm(temperature=0.4, cache=False if has_image_context else None)

    try:
        response = await llm.ainvoke(conversation, config=config)
        resolution_text = str(response.content).strip()
        new_messages = [AIMessage(content=resolution_text)]
        error = None
        # Only cache when the FAQ search actually returned chunks.
        # faq_chunks=[] means Qdrant was unavailable or found nothing, so the
        # LLM had no real context and likely produced an apology/fallback message.
        # 4) Write back to application-level cache, only if no image context was present
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
