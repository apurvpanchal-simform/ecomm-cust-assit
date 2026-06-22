"""
Agent node responsible for maintaining a compact, rolling summary of the conversation.
"""

import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import observe
from pydantic import BaseModel, Field

from app.graph.state import AgentState
from app.graph.utils import filter_tool_messages
from app.services.llm_factory import get_llm

logger = logging.getLogger(__name__)

SUMMARIZER_PROMPT = """You are a conversation summarizer. Your job is to extract and update structured conversation facts and issue tracking from the conversation history.

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


class ActiveIssue(BaseModel):
    """ActiveIssue model representing a currently open issue with its details and status.

    Inherits from BaseModel and includes fields such as id, title, description, and timestamps.
    """

    issue_description: str = Field(
        description="Description of the active unresolved issue/goal."
    )
    turns_active: int = Field(
        description="Number of consecutive turns the issue has remained unresolved."
    )


class StructuredSummary(BaseModel):
    """Structured summary model containing a title, list of bullet points, and optional metadata, with validation and serialization provided by BaseModel."""

    customer_profile_and_preferences: list[str] = Field(
        description="List of user preferences such as sizes, color choices, budget limits, styles."
    )
    mentioned_orders: list[str] = Field(
        description="List of specific order IDs (lowercase), tracking numbers, or refund states mentioned."
    )
    active_issues: list[ActiveIssue] = Field(
        description="List of active unresolved issues/goals."
    )
    resolved_issues: list[str] = Field(
        description="List of issues/questions that have been successfully answered or completed."
    )
    escalate_to_human: bool = Field(
        description="Set to true ONLY IF the user explicitly asks for a human agent AND there is an active unresolved issue. Do not escalate if there are no active issues."
    )


def _format_summary_to_markdown(summary: StructuredSummary) -> str:
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


@observe(name="summarizer_node")
async def summarizer_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Updates the running summary of the conversation if a sufficient chunk of new messages exists.

    To manage token limits and maintain state relevance, this node condenses older conversation
    history into a dense list of facts (preferences, constraints, entity IDs). It operates on a
    sliding window approach, leaving recent messages unsummarized to preserve immediate context.

    Args:
        state: The global AgentState containing the current message history and summary.
        config: Execution configuration.

    Returns:
        A dictionary containing the updated `chat_summary` and the new `summarized_message_count`.
    """
    all_messages = state.get("messages", [])
    summarized_count = state.get("summarized_message_count", 0)
    current_summary = state.get("chat_summary", "")

    WINDOW_SIZE = 4
    CHUNK_SIZE = 4

    # We want to keep the last WINDOW_SIZE messages completely unsummarized.
    # To prevent "telephone" effect, we wait until we have CHUNK_SIZE extra messages
    # before we run the summarizer.

    # HEURISTIC CHECK: Always check the unsummarized messages for immediate escalation or resolution.
    # We search backwards to find the MOST RECENT event.
    unsummarized_messages = all_messages[summarized_count:]
    heuristic_escalate = None
    for msg in reversed(unsummarized_messages):
        content = getattr(msg, "content", "") or ""
        msg_type = getattr(msg, "type", "")
        
        # 1. Did the system recently resolve it?
        if msg_type == "system" and "human support agent has resolved" in content.lower():
            logger.info("[SUMMARIZER] Heuristic override: SYSTEM resolution detected → False")
            heuristic_escalate = False
            break
            
        # 2. Did the AI recently offer human support?
        if msg_type == "ai":
            lower_content = content.lower()
            trigger_phrases = ["contact support", "human agent", "live agent", "support team", "help desk", "representative"]
            if any(p in lower_content for p in trigger_phrases):
                logger.info("[SUMMARIZER] Heuristic override: AI offered support → True")
                heuristic_escalate = True
                break

    # If the heuristic wants to escalate, we verify there is actually an active unresolved issue.
    # If there are no active issues in the history summary AND the AI did not explicitly
    # confirm connection/escalation in the current turn, we block the escalation.
    if heuristic_escalate:
        has_active_issues = False
        if current_summary:
            import re
            match = re.search(r"### Active Issues\n(.*?)(?:\n###|$)", current_summary, re.DOTALL)
            if match:
                block = match.group(1).strip()
                if block and "- None" not in block:
                    has_active_issues = True
        if not has_active_issues:
            for msg in unsummarized_messages:
                if getattr(msg, "type", "") == "ai":
                    content_lower = (msg.content or "").lower()
                    if any(phrase in content_lower for phrase in ["connecting you", "escalating this", "representative will be", "transferring you"]):
                        has_active_issues = True
                        break
        if not has_active_issues:
            logger.info("[SUMMARIZER] Heuristic requested escalation, but blocked because no active issues were found.")
            heuristic_escalate = False

    if len(all_messages) - summarized_count >= WINDOW_SIZE + CHUNK_SIZE:
        messages_to_grab = CHUNK_SIZE
        messages_to_summarize = all_messages[
            summarized_count : summarized_count + messages_to_grab
        ]

        # Format these messages into a readable string
        formatted_messages = []
        for msg in filter_tool_messages(messages_to_summarize):
            msg_type = getattr(msg, "type", "")
            role = "User" if msg_type == "human" else "Assistant"
            content = msg.content
            if content:
                formatted_messages.append(f"{role}: {content}")

        new_content_text = "\n\n".join(formatted_messages)

        llm = get_llm(temperature=0.0)
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

            # The LLM's opinion is used only if the heuristic didn't trigger
            escalate_flag = structured_summary.escalate_to_human
            if heuristic_escalate is not None:
                escalate_flag = heuristic_escalate
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

            logger.info(
                f"\n========== NEW CHAT SUMMARY ==========\n{new_summary}\n======================================\n"
            )

            try:
                os.makedirs("data", exist_ok=True)
                with open("data/summaries.log", "a", encoding="utf-8") as f:
                    f.write(f"========== SUMMARY ==========\n{new_summary}\n\n")
            except Exception as e:
                logger.warning("Could not write to summaries.log: %s", e)

            # The new summarized count should include all messages we just summarized
            new_summarized_count = summarized_count + len(messages_to_summarize)

            return {
                "chat_summary": new_summary,
                "summarized_message_count": new_summarized_count,
                "escalate_to_human": escalate_flag,
            }
        except Exception as e:
            logger.error("Failed to generate summary: %s", e)
            state_update = {}
            if heuristic_escalate is not None:
                state_update["escalate_to_human"] = heuristic_escalate
            return state_update
    else:
        # If no summarization is needed, apply heuristic if found
        state_update = {}
        if heuristic_escalate is not None:
            state_update["escalate_to_human"] = heuristic_escalate

        logging.getLogger(__name__).debug(
            f"[SUMMARIZER] Sleeping. Total msgs: {len(all_messages)}, Summarized: {summarized_count}"
        )
        return state_update
