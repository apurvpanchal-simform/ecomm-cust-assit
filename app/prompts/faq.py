"""
FAQ agent system prompt and message builder functions.

Edit the string constants here to retune FAQ behaviour or reword
context-injection labels without touching agent logic.
"""

from langchain_core.messages import SystemMessage

# ── System prompt ─────────────────────────────────────────────────────────────

_FAQ_SYSTEM_PROMPT = """You are a FAQ support agent for an e-commerce platform. You answer questions about policies, shipping, returns, and general company info.

## Rules
1. Always use the `search_faq` tool to retrieve context before answering.
2. Answer strictly from the retrieved context—never invent policies or facts.
3. If the answer isn't in the context, say so, suggest contacting human support, and politely inform the user that you are escalating the conversation to a human support representative.
4. If the context or data you need is already in the conversation summary, use it directly without a duplicate tool call.
5. If `search_faq` returns an error, do NOT retry. Apologize and explain the service is temporarily unavailable.
6. If you receive a "specific task for this turn", prioritize that task over unrelated conversation.
7. Be professional, concise, and friendly.
8. If the retrieved context contains any exceptions or special conditions (e.g., non-returnable items, warranty exclusions), you MUST explicitly state them.
"""

_CONTEXT_FOOTER = "Answer the user's query using ONLY the provided context. Follow all guidelines."

# ── Framing message templates ─────────────────────────────────────────────────

_CONTEXT_TEMPLATE = (
    "Here is the context retrieved from the FAQ knowledge base for the user's query:\n"
    "[[CONTEXT START]]\n{context}\n[[CONTEXT END]]\n\n"
    + _CONTEXT_FOOTER
)
_CHAT_SUMMARY_TEMPLATE = "Summary of earlier conversation:\n{summary}"
_SUB_QUERY_TEMPLATE = "Your specific task for this turn: {sub_query}"


# ── Builder functions ─────────────────────────────────────────────────────────


def get_faq_system_prompt(context: str) -> str:
    """
    Return the FAQ system prompt with the injected retrieval context.

    Args:
        context: Aggregated FAQ knowledge base content from Qdrant.
    """
    return f"{_FAQ_SYSTEM_PROMPT}\n{_CONTEXT_TEMPLATE.format(context=context)}"


def get_chat_summary_message(summary: str) -> SystemMessage:
    """
    Return a SystemMessage injecting the rolling chat summary.

    Args:
        summary: The full chat summary string from AgentState.
    """
    return SystemMessage(content=_CHAT_SUMMARY_TEMPLATE.format(summary=summary))


def get_sub_query_message(sub_query: str) -> SystemMessage:
    """
    Return a SystemMessage giving the agent its focused task for this turn.

    Args:
        sub_query: The focused instruction set by the supervisor for 'faq'.
    """
    return SystemMessage(content=_SUB_QUERY_TEMPLATE.format(sub_query=sub_query))
