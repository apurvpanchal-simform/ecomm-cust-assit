"""
Shared utility helpers for processing LangChain message lists in graph nodes.

These helpers solve two recurring problems:
1. Some LLM providers (e.g. Groq) reject conversation history that contains
   `ToolMessage` objects or `AIMessage` objects with `tool_calls` when the LLM
   is not bound with `.bind_tools()`.  `filter_tool_messages()` safely strips
   those before they reach non-tool-using nodes.

2. Message content can be either a plain string or a list of typed content
   blocks (e.g. Anthropic multimodal format).  `get_message_text()` normalises
   both representations into a single string.
"""

from langchain_core.messages import AIMessage


def filter_tool_messages(messages: list) -> list:
    """
    Remove tool-related messages that would break non-tool-using LLMs.

    Groq (and some other providers) raise ``Tools should have a name!`` if the
    conversation history contains ``AIMessage`` objects with ``tool_calls`` or
    any ``ToolMessage`` objects when the model was not configured with
    ``.bind_tools()``.

    Call this helper before passing ``recent_messages`` to any node that does
    **not** use tools — i.e., supervisor, faq, synthesizer, and summarizer.

    Filtering rules:
    - ``ToolMessage`` objects are dropped entirely.
    - ``AIMessage`` objects that contain *only* tool_calls (no text content)
      are dropped.
    - ``AIMessage`` objects that have both text content **and** tool_calls are
      kept, but the ``tool_calls`` attribute is stripped so the text context
      is preserved without triggering provider errors.
    - All other message types are passed through unchanged.

    Args:
        messages: Raw list of LangChain messages from the graph state.

    Returns:
        A filtered list safe to pass to a non-tool-using LLM.
    """
    filtered: list = []
    for msg in messages:
        msg_type = getattr(msg, "type", "")

        # Drop standalone ToolMessages entirely — they reference tool call IDs
        # that the non-tool LLM has no context for.
        if msg_type == "tool":
            continue

        # Handle AIMessages that carry tool_calls: keep the message but set its
        # content to the tool name(s) only, stripping the tool_calls attribute.
        if msg_type == "ai" and getattr(msg, "tool_calls", None):
            tool_names = [t.get("name") for t in msg.tool_calls if t.get("name")]
            content = ", ".join(tool_names)
            filtered.append(AIMessage(content=content))
            continue

        filtered.append(msg)

    return filtered


def get_message_text(msg) -> str:
    """
    Extract the plain text content from a LangChain message.

    Handles both simple string content and the list-of-blocks format used by
    multimodal message types (e.g. Anthropic content blocks with ``type: "text"``).

    Args:
        msg: A LangChain message object or a plain dict with a ``content`` key.

    Returns:
        A single concatenated string.  Returns an empty string if the message
        has no text content.
    """
    # Support both LangChain message objects and plain dicts
    if isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        content = getattr(msg, "content", "") or ""

    if isinstance(content, list):
        # Multimodal format: list of typed content blocks
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts)

    # Simple string content
    return str(content)


def filter_ai_messages(messages: list) -> list:
    """
    Filter and return only the AI/assistant messages from a list of messages.

    Args:
        messages: A list of LangChain message objects.

    Returns:
        A list of AIMessage objects.
    """
    return [
        msg
        for msg in messages
        if getattr(msg, "type", "") == "ai" or isinstance(msg, AIMessage)
    ]
