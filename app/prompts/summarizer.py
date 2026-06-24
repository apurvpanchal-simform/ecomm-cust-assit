"""
Summarizer agent system prompt and message builder functions.

Edit the string constants here to retune summarization behaviour or reword
the framing messages without touching agent logic.
"""

from langchain_core.messages import HumanMessage, SystemMessage

# ── System prompt ─────────────────────────────────────────────────────────────

_SUMMARIZER_PROMPT = """You are a conversation summarizer. Your job is to extract and update structured conversation facts and issue tracking from the conversation history.

Analyze the 'Previous Summary' (if provided) and the 'New messages to incorporate into the summary'. Then populate the StructuredSummary schema according to these rules:

1. **Merge & Update**: If a Previous Summary is provided, merge its `customer_profile_and_preferences`, `mentioned_orders`, and `resolved_issues` with the new information extracted from the new messages. Do not discard previous facts unless they are explicitly outdated, resolved, or overridden.
2. **Customer Profile & Preferences**: Extract and maintain user preferences (sizes, color choices, budget limits, styles, materials) crucial for context.
3. **Mentioned Orders**: Extract and maintain specific order IDs (always format in lowercase), tracking numbers, or refund states.
4. **Active Issues Tracking**:
   - Track active unresolved customer issues or goals (e.g., return query, order tracking request, refund dispute).
   - If an active issue from the Previous Summary is still unresolved in the new messages, increment its `turns_active` counter by 2 (since the summarizer runs once every 2 turns).
   - If a new issue is introduced in the new messages, add it to `active_issues` with `turns_active` initialized to 2.
   - If an issue is resolved in the new messages, DO NOT list it in `active_issues`; instead, move it to `resolved_issues`.
5. **Resolved Issues**: List issues or questions that have been successfully resolved, answered, or completed.
6. **Escalate to Human**: Set `escalate_to_human` to `True` IF either of these conditions are met: (a) the user explicitly asks to speak to a human/agent/support AND there is at least one unresolved active issue in `active_issues`; OR (b) the AI Assistant's response advises the user to contact the support team, help desk, or a human agent. CRITICAL: DO NOT set escalate_to_human to True for the same resolved issue if the conversation history shows that a human support agent recently resolved that issue (e.g. indicated by a SYSTEM message).
"""

# ── Framing message templates ─────────────────────────────────────────────────

_PREVIOUS_SUMMARY_TEMPLATE = "Previous Summary:\n{summary}"
_NEW_MESSAGES_TEMPLATE = "New messages to incorporate into the summary:\n{content}"


# ── Builder functions ─────────────────────────────────────────────────────────


def get_summarizer_prompt() -> str:
    """Return the summarizer's system prompt string."""
    return _SUMMARIZER_PROMPT


def get_previous_summary_message(summary: str) -> SystemMessage:
    """
    Return a SystemMessage carrying the previous structured summary.

    Args:
        summary: The existing summary string from AgentState.
    """
    return SystemMessage(content=_PREVIOUS_SUMMARY_TEMPLATE.format(summary=summary))


def get_new_messages_input(content: str) -> HumanMessage:
    """
    Return a HumanMessage containing the new conversation text to be summarized.

    Args:
        content: Formatted string of recent messages to incorporate.
    """
    return HumanMessage(content=_NEW_MESSAGES_TEMPLATE.format(content=content))
