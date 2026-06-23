"""
Agent node responsible for maintaining a compact, rolling summary of the conversation.

`summarizer_node` runs after every conversation turn (after the synthesizer).
Its two responsibilities are:
1. **Escalation detection**: Scan recent messages for signals that the user needs
   a human agent, using a fast heuristic (no LLM call required).
2. **Conversation compression**: When enough new messages have accumulated, call
   an LLM to compress the oldest unsummarised messages into a structured
   `StructuredSummary` and store it in `chat_summary`.

Sliding window strategy
-----------------------
We always keep the last WINDOW_SIZE=4 messages fully unsummarised so downstream
agents always have immediate context.  Summarisation is triggered only when at
least CHUNK_SIZE=4 more messages have accumulated beyond the window, preventing
the "telephone game" effect of summarising-too-often.
"""

import logging
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import observe

from app.schemas.summary import ActiveIssue, StructuredSummary

from app.config.llm_config import SUMMARIZER_LLM_CONFIG
from app.graph.state import AgentState
from app.graph.utils import filter_ai_messages, filter_tool_messages, get_message_text
from app.prompts import SUMMARIZER_PROMPT
from app.services.llm_factory import get_llm

logger = logging.getLogger(__name__)

# Sliding window constants
_WINDOW_SIZE = 4  # Always keep this many recent messages unsummarised
_CHUNK_SIZE = 4  # Wait for this many new messages before summarising


# ── Private helpers ───────────────────────────────────────────────────────────


def _format_summary_to_markdown(summary: StructuredSummary) -> str:
    """
    Convert a StructuredSummary Pydantic object into a human-readable markdown string.

    The output is stored in `state["chat_summary"]` and injected into agent prompts
    as background context on subsequent conversation turns.

    Args:
        summary: The structured summary produced by the LLM.

    Returns:
        A markdown-formatted string with section headers for each summary field.
    """
    lines = []

    lines.append("### Customer Profile & Preferences")
    if summary.customer_profile_and_preferences:
        for pref in summary.customer_profile_and_preferences:
            lines.append(f"- {pref}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("### Mentioned Orders")
    if summary.mentioned_orders:
        for order in summary.mentioned_orders:
            lines.append(f"- {order}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("### Active Issues")
    if summary.active_issues:
        for issue in summary.active_issues:
            lines.append(
                f"- [Turns Active: {issue.turns_active}] {issue.issue_description}"
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.append("### Resolved Issues")
    if summary.resolved_issues:
        for issue in summary.resolved_issues:
            lines.append(f"- {issue}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("### Escalate to Human")
    lines.append(f"- {summary.escalate_to_human}")

    return "\n".join(lines)


# SYSTEM message substrings that confirm the conversation was resolved
_RESOLUTION_SYSTEM_PHRASES = [
    "human support agent has resolved",
    "automated ai assistant is now re-enabled",  # auto-timeout resolve phrase
]


def _detect_escalation_heuristic(
    unsummarized_messages: list,
) -> bool | None:
    """
    Scan recent messages for escalation signals without calling the LLM.

    Walks the unsummarised messages in reverse order so the most recent event
    takes precedence.  Returns a boolean override, or None if no clear signal
    is found (in which case the LLM's judgment is used).

    Detection logic (checked in order, most-recent message wins):
    1. SYSTEM message confirming resolution (human agent or auto-timeout) → False
    2. AI message containing support-related phrases → True (escalate)

    Args:
        unsummarized_messages: Messages not yet included in the rolling summary.

    Returns:
        True if a heuristic escalation trigger was found,
        False if a heuristic resolution trigger was found,
        None if no clear signal was detected.
    """
    _AI_ESCALATION_PHRASES = [
        "contact support",
        "human agent",
        "live agent",
        "support team",
        "help desk",
        "representative",
    ]

    for msg in reversed(unsummarized_messages):
        content = get_message_text(msg)
        msg_type = getattr(msg, "type", "")

        # Highest priority: any SYSTEM message confirming resolution
        if msg_type == "system":
            content_lower = content.lower()
            if any(phrase in content_lower for phrase in _RESOLUTION_SYSTEM_PHRASES):
                logger.info(
                    "[SUMMARIZER] Heuristic override: SYSTEM resolution detected → False"
                )
                return False

        # Secondary: AI offered to connect the user with human support
        if msg_type == "ai":
            lower_content = content.lower()
            if any(phrase in lower_content for phrase in _AI_ESCALATION_PHRASES):
                logger.info(
                    "[SUMMARIZER] Heuristic override: AI offered support → True"
                )
                return True

    return None  # No clear signal — defer to LLM


def _is_recently_resolved(current_summary: str, unsummarized_messages: list) -> bool:
    """
    Return True if this conversation was recently resolved by a human agent
    or by the auto-timeout mechanism.

    Checks two signals:
    1. A SYSTEM message in recent (unsummarized) messages that confirms resolution.
    2. The chat_summary's "Resolved Issues" section is non-empty (i.e. has past
       resolutions), which means a human agent has historically handled this chat.

    Args:
        current_summary: The current rolling markdown summary string.
        unsummarized_messages: Recent unsummarized messages.

    Returns:
        True if there is a clear resolution signal that should block re-escalation.
    """
    # Signal 1 — SYSTEM message in current turn window
    for msg in unsummarized_messages:
        if getattr(msg, "type", "") == "system":
            content_lower = get_message_text(msg).lower()
            if any(phrase in content_lower for phrase in _RESOLUTION_SYSTEM_PHRASES):
                return True

    # Signal 2 — chat_summary has a non-empty Resolved Issues section
    if current_summary:
        match = re.search(
            r"### Resolved Issues\n(.*?)(?:\n###|$)", current_summary, re.DOTALL
        )
        if match:
            block = match.group(1).strip()
            if block and "- None" not in block:
                logger.info(
                    "[SUMMARIZER] Resolved Issues found in summary — "
                    "blocking any re-escalation."
                )
                return True

    return False


def _verify_escalation_has_active_issues(
    heuristic_escalate: bool,
    current_summary: str,
    unsummarized_messages: list,
) -> bool:
    """
    Prevent false-positive escalations when there are no unresolved issues.

    The heuristic fires on any mention of "support" or "human agent" — but we
    should not escalate if the conversation has no active unresolved issues.
    This function checks the current summary and recent messages to verify that
    at least one active issue actually exists.

    If the summary's Active Issues section is "- None" AND no explicit
    escalation/connection message appears in the recent AI messages, we block
    the heuristic escalation.

    Args:
        heuristic_escalate: Whether the heuristic wants to escalate.
        current_summary: The current rolling markdown summary string.
        unsummarized_messages: Recent unsummarised messages.

    Returns:
        The (possibly corrected) escalation flag.
    """
    if not heuristic_escalate:
        return heuristic_escalate  # Only need to verify positive signals

    has_active_issues = False

    # Check the summary for non-empty Active Issues section
    if current_summary:
        match = re.search(
            r"### Active Issues\n(.*?)(?:\n###|$)", current_summary, re.DOTALL
        )
        if match:
            block = match.group(1).strip()
            if block and "- None" not in block:
                has_active_issues = True

    # Also check for explicit escalation confirmation in recent AI messages
    if not has_active_issues:
        _ESCALATION_CONFIRM_PHRASES = [
            "connecting you",
            "escalating this",
            "representative will be",
            "transferring you",
        ]
        for msg in filter_ai_messages(unsummarized_messages):
            content_lower = get_message_text(msg).lower()
            if any(phrase in content_lower for phrase in _ESCALATION_CONFIRM_PHRASES):
                has_active_issues = True
                break

    if not has_active_issues:
        logger.info(
            "[SUMMARIZER] Heuristic requested escalation, but blocked because no active issues were found."
        )
        return False

    return True


def _apply_heuristic_override(
    new_summary: str, heuristic_escalate: bool | None, llm_escalate: bool
) -> tuple[str, bool]:
    """
    Apply the heuristic escalation flag to override the LLM's decision if needed.

    When the heuristic detected a clear signal, it takes precedence over the LLM.
    The summary text is also patched to match the override so the state stays consistent.

    Args:
        new_summary: The markdown summary string generated by the LLM.
        heuristic_escalate: The heuristic's override (True/False) or None (no override).
        llm_escalate: The LLM's own escalation decision from the StructuredSummary.

    Returns:
        A tuple of (updated_summary_string, final_escalation_flag).
    """
    if heuristic_escalate is None:
        # No override — use the LLM's decision as-is
        return new_summary, llm_escalate

    escalate_flag = heuristic_escalate

    # Patch the summary text to match the override
    if not escalate_flag:
        new_summary = new_summary.replace(
            "### Escalate to Human\n- True",
            "### Escalate to Human\n- False",
        )
    else:
        new_summary = new_summary.replace(
            "### Escalate to Human\n- False",
            "### Escalate to Human\n- True",
        )

    return new_summary, escalate_flag


def _get_messages_to_summarize(all_messages: list, summarized_count: int) -> list:
    """
    Select the next chunk of messages to be condensed into the rolling summary.

    Takes exactly `_CHUNK_SIZE` messages starting from `summarized_count`,
    leaving the most recent `_WINDOW_SIZE` messages outside the summary window.

    Args:
        all_messages: The full message history from the state.
        summarized_count: How many messages are already included in the summary.

    Returns:
        A list of `_CHUNK_SIZE` messages to summarize.
    """
    return all_messages[summarized_count : summarized_count + _CHUNK_SIZE]


def _log_summary(new_summary: str) -> None:
    """
    Log the generated summary to the console and append it to `data/summaries.log`.

    The file log is best-effort — a failure is logged as a warning and does not
    prevent the summary from being stored in the graph state.

    Args:
        new_summary: The formatted markdown summary string.
    """
    logger.info(
        "\n========== NEW CHAT SUMMARY ==========\n%s\n======================================\n",
        new_summary,
    )
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/summaries.log", "a", encoding="utf-8") as f:
            f.write(f"========== SUMMARY ==========\n{new_summary}\n\n")
    except Exception as e:
        logger.warning("Could not write to summaries.log: %s", e)


# ── Node ──────────────────────────────────────────────────────────────────────


@observe(name="summarizer_node")
async def summarizer_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Update the rolling conversation summary and detect human escalation signals.

    Flow:
    1. Run the heuristic escalation scan on all unsummarised messages (no LLM needed).
    2. If the heuristic wants to escalate, verify that there are actually active issues.
    3. Check whether enough new messages have accumulated to trigger summarisation.
    4. If yes: select the next chunk, call the LLM, apply heuristic override, log and return.
    5. If no:  return the heuristic escalation flag (if any) without summarising.

    Args:
        state: The global AgentState.
        config: LangChain execution configuration.

    Returns:
        A partial state dict.  May include:
            chat_summary             — updated markdown summary (if summarised)
            summarized_message_count — new count of summarised messages
            escalate_to_human        — escalation flag (if heuristic or LLM set it)
        Returns an empty dict if nothing changed.
    """
    all_messages = state.get("messages", [])
    summarized_count = state.get("summarized_message_count", 0)
    current_summary = state.get("chat_summary", "")
    unsummarized_messages = all_messages[summarized_count:]

    # ── 1. Run heuristic escalation scan ─────────────────────────────────────
    heuristic_escalate = _detect_escalation_heuristic(unsummarized_messages)

    # ── 2. Verify heuristic escalation has backing active issues ─────────────
    if heuristic_escalate:
        heuristic_escalate = _verify_escalation_has_active_issues(
            heuristic_escalate, current_summary, unsummarized_messages
        )

    # ── 2b. Block re-escalation on already-resolved conversations ────────────
    # If a human agent (or the auto-timeout) recently resolved this conversation,
    # never allow the summarizer to set escalate_to_human=True again for the
    # same underlying issue. The resolution SYSTEM message or a non-empty
    # Resolved Issues section in the summary is the authoritative signal.
    recently_resolved = _is_recently_resolved(current_summary, unsummarized_messages)
    if recently_resolved and heuristic_escalate:
        logger.info(
            "[SUMMARIZER] Blocking heuristic escalation: conversation was recently resolved."
        )
        heuristic_escalate = False

    # ── 3. Check if enough messages have accumulated to summarise ─────────────
    # We need at least WINDOW_SIZE + CHUNK_SIZE unsummarised messages before we
    # compress any of them, to ensure the window always stays fully unsummarised.
    if len(all_messages) - summarized_count < _WINDOW_SIZE + _CHUNK_SIZE:
        # Not enough messages — just apply heuristic escalation if triggered
        logger.debug(
            "[SUMMARIZER] Sleeping. Total msgs: %d, Summarized: %d",
            len(all_messages),
            summarized_count,
        )
        state_update: dict = {}
        if heuristic_escalate is not None:
            state_update["escalate_to_human"] = heuristic_escalate
        return state_update

    # ── 4. Select the messages to summarise ───────────────────────────────────
    messages_to_summarize = _get_messages_to_summarize(all_messages, summarized_count)

    # Format selected messages as "User: ..." / "Assistant: ..." pairs for the LLM
    formatted_messages = []
    for msg in filter_tool_messages(messages_to_summarize):
        msg_type = getattr(msg, "type", "")
        role = "User" if msg_type == "human" else "Assistant"
        content_str = get_message_text(msg)
        if content_str:
            formatted_messages.append(f"{role}: {content_str}")

    new_content_text = "\n\n".join(formatted_messages)

    # ── 5. Call the LLM to generate the structured summary ────────────────────
    llm = get_llm(
        temperature=SUMMARIZER_LLM_CONFIG.temperature,
        cache=SUMMARIZER_LLM_CONFIG.default_cache,
    )
    structured_llm = llm.with_structured_output(StructuredSummary)

    prompt_messages = [SystemMessage(content=SUMMARIZER_PROMPT)]
    if current_summary:
        prompt_messages.append(
            SystemMessage(content=f"Previous Summary:\n{current_summary}")
        )
    prompt_messages.append(
        HumanMessage(
            content=f"New messages to incorporate into the summary:\n{new_content_text}"
        )
    )

    try:
        structured_summary = await structured_llm.ainvoke(
            prompt_messages, config=config
        )
        new_summary = _format_summary_to_markdown(structured_summary)

        # ── 6. Apply heuristic override (takes precedence over LLM) ──────────
        new_summary, escalate_flag = _apply_heuristic_override(
            new_summary=new_summary,
            heuristic_escalate=heuristic_escalate,
            llm_escalate=structured_summary.escalate_to_human,
        )

        # ── 6b. Final guard: block LLM re-escalation on resolved conversations ─
        if escalate_flag and recently_resolved:
            logger.info(
                "[SUMMARIZER] Blocking LLM escalation: conversation was recently resolved."
            )
            escalate_flag = False
            new_summary = new_summary.replace(
                "### Escalate to Human\n- True",
                "### Escalate to Human\n- False",
            )

        # ── 7. Log and persist the new summary ────────────────────────────────
        _log_summary(new_summary)

        new_summarized_count = summarized_count + len(messages_to_summarize)

        return {
            "chat_summary": new_summary,
            "summarized_message_count": new_summarized_count,
            "escalate_to_human": escalate_flag,
        }

    except Exception as e:
        logger.error("Failed to generate summary: %s", e)
        # Still apply heuristic even when LLM fails
        state_update = {}
        if heuristic_escalate is not None:
            state_update["escalate_to_human"] = heuristic_escalate
        return state_update
