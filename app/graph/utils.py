"""Shared utility helpers for graph message processing."""

from langchain_core.messages import AIMessage


def filter_tool_messages(messages: list) -> list:
    """Filter out tool-related messages that would break LLMs not bound with tools.

    Groq (and some other providers) will error with
    ``Tools should have a name!`` if the conversation history contains
    ``AIMessage`` objects with ``tool_calls`` or ``ToolMessage`` objects,
    but the LLM was not configured with ``.bind_tools()``.

    Call this helper before sending ``recent_messages`` to any LLM node
    that does **not** use tools (supervisor, faq, filter_extractor,
    synthesizer, summarizer).

    For AIMessages that have *both* text content and tool_calls, we keep
    the text content but strip the tool_calls so the conversational
    context is preserved.
    """
    filtered: list = []
    for msg in messages:
        msg_type = getattr(msg, "type", "")

        # Drop standalone ToolMessages entirely
        if msg_type == "tool":
            continue

        # Handle AIMessages that carry tool_calls
        if msg_type == "ai" and getattr(msg, "tool_calls", None):
            # If there is meaningful text content alongside the tool call,
            # keep a sanitised copy so context is not lost.
            content = msg.content
            if content and str(content).strip():
                filtered.append(
                    AIMessage(
                        content=content,
                    )
                )
            # Otherwise just drop the message (it was a pure tool-call)
            continue

        filtered.append(msg)

    return filtered
