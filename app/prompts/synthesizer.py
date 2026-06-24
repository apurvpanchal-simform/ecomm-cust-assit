"""
Synthesizer agent system prompt and message builder functions.

Edit the string constants here to retune how the synthesizer frames the
merged agent responses without touching agent logic.
"""

from langchain_core.messages import HumanMessage, SystemMessage

# ── System prompt ─────────────────────────────────────────────────────────────

_SYNTHESIZER_PROMPT = """You are an e-commerce assistant. You receive raw responses from multiple backend agents that ran during a single conversation turn.

## Task
Merge all agent responses into one cohesive, natural reply for the user.

## Rules
1. Combine information naturally—don't mechanically list each agent's output.
2. If one response is a refusal (e.g., "I can't help with coding"), weave it in gracefully alongside valid information.
3. Never invent facts—use only what the agents provided.
4. Keep the tone warm and helpful.
5. NEVER use inline code (backticks `) to format labels or monetary amounts (e.g., do NOT write `Subtotal: \\`$10\\`` or `\\`**Tax**\\``). Use standard bold text instead.
6. If any agent's response indicates that the conversation is being escalated/transferred to a human agent, ensure the synthesized response clearly confirms to the user that they are being connected to a human representative.
"""

# ── Framing message templates ─────────────────────────────────────────────────

_AGENT_RESPONSES_HEADER = "Raw Agent Responses to combine:"
_AGENT_RESPONSE_LINE_TEMPLATE = "- [{source}]: {text}"


# ── Builder functions ─────────────────────────────────────────────────────────


def get_synthesizer_prompt() -> str:
    """Return the synthesizer's system prompt string."""
    return _SYNTHESIZER_PROMPT


def get_synthesizer_system_message() -> SystemMessage:
    """Return the synthesizer's system prompt as a SystemMessage."""
    return SystemMessage(content=_SYNTHESIZER_PROMPT)


def get_agent_responses_message(agent_messages: list) -> HumanMessage:
    """
    Format all agent response messages into a single HumanMessage for the synthesizer.

    Args:
        agent_messages: List of AIMessage objects from downstream agents.
                        Each message may optionally have a `name` attribute set
                        by the agent node to identify the source.
    """
    from app.graph.utils import get_message_text  # avoid circular imports

    lines = [_AGENT_RESPONSES_HEADER]
    for i, msg in enumerate(agent_messages):
        source = getattr(msg, "name", f"Agent_{i + 1}")
        lines.append(_AGENT_RESPONSE_LINE_TEMPLATE.format(source=source, text=get_message_text(msg)))
    return HumanMessage(content="\n".join(lines))
