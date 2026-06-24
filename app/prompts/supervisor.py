"""
Supervisor agent system prompt and message builder functions.

Edit the string constants here to retune the supervisor's routing behaviour
without touching any agent logic.
"""

from langchain_core.messages import SystemMessage

# ── System prompt ─────────────────────────────────────────────────────────────

_SUPERVISOR_SYSTEM_PROMPT = """You are a routing supervisor for an e-commerce support system. You analyze the user's message and delegate work to the right agents.

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
7. If the user asks to speak to a human support agent, representative, live support, or wants to escalate their issue, only route to a downstream agent (like order or faq) if they mention a specific issue (e.g., a broken item, refund, tracking). If they do not mention any specific issue, return empty `pending_agents`, and in `response` politely explain that you can connect them to a human agent once they describe the specific issue they need help with.

## Execution Rules
8. Order `pending_agents` by dependency: if Agent B needs Agent A's output, list A first.
9. Provide a precise `sub_queries` entry for every agent you trigger, telling it exactly what to do this turn.

## Search Filters (when routing to `image_search_agent`)
10. Extract a clean `search_query` combining user intent and image description. Exclude price terms from the query.
11. Set `min_price` / `max_price` if the user mentions a budget (e.g., "under $50" → max_price=50).

## Synthesis
A downstream synthesizer will merge all agent responses with your `response` field into one reply for the user. Write naturally.
"""

# ── Framing message templates ─────────────────────────────────────────────────

_CHAT_SUMMARY_TEMPLATE = "Background Context (Older info, lower priority):\n{summary}"
_LATEST_CONVO_HEADER = "Latest Conversation (Highest priority):"
_IMAGE_SAFE_INSTRUCTION = (
    "\nIMPORTANT: This image is SAFE and approved. You MUST route to "
    "'image_search_agent'. DO NOT apologize or claim it was blocked due to policy."
)


# ── Builder functions ─────────────────────────────────────────────────────────


def get_supervisor_system_prompt() -> str:
    """Return the supervisor's routing system prompt."""
    return _SUPERVISOR_SYSTEM_PROMPT


def get_chat_summary_message(summary: str) -> SystemMessage:
    """
    Return a SystemMessage injecting the rolling chat summary as background context.

    The summary is capped at 800 chars to avoid prompt bloat.

    Args:
        summary: The full chat summary string from AgentState.
    """
    if len(summary) > 800:
        summary = "[Truncated earlier summary]...\n" + summary[-800:]
    return SystemMessage(content=_CHAT_SUMMARY_TEMPLATE.format(summary=summary))


def get_image_context_message(
    image_desc: str | None,
    image_tags: list[str] | None,
    is_safe: bool,
) -> SystemMessage:
    """
    Return a SystemMessage describing the uploaded image for the supervisor.

    Args:
        image_desc: Textual description / OCR result from the vision service.
        image_tags: List of classification tags.
        is_safe: Whether the image passed the safety check.
    """
    lines = ["The user uploaded an image. Image Analysis:"]
    if image_desc:
        lines.append(f"Description/OCR: {image_desc}")
    if image_tags:
        lines.append(f"Tags: {', '.join(image_tags)}")
    if is_safe:
        lines.append(_IMAGE_SAFE_INSTRUCTION)
    return SystemMessage(content="\n".join(lines))


def get_image_safety_warning_message(warning: str) -> SystemMessage:
    """
    Return a SystemMessage carrying an ephemeral image safety warning.

    Args:
        warning: The safety warning text set by the image analyzer node.
    """
    return SystemMessage(content=f"SYSTEM: {warning}")


def get_latest_conversation_header() -> SystemMessage:
    """Return the header SystemMessage that precedes recent conversation turns."""
    return SystemMessage(content=_LATEST_CONVO_HEADER)
