"""
Agent node that synthesizes outputs from multiple sub-agents into a single reply.

`synthesizer_node` runs after all sub-agents (FAQ, Order, Image Search) have
completed their work.  It is a no-op if only one agent ran (its response is
already user-ready).  When multiple agents ran, it uses an LLM to merge their
individual responses into one cohesive, natural reply.

Note: The synthesizer runs even on single-agent turns (it just returns {}),
so the graph topology remains uniform regardless of how many agents were invoked.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import observe

from app.config.llm_config import SYNTHESIZER_LLM_CONFIG
from app.graph.state import AgentState
from app.graph.utils import filter_ai_messages, filter_tool_messages, get_message_text
from app.prompts import get_synthesizer_system_message, get_agent_responses_message
from app.services.llm_factory import get_llm

logger = logging.getLogger(__name__)


@observe(name="synthesizer_node")
async def synthesizer_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Merge multiple sub-agent responses into a single cohesive user-facing reply.

    Flow:
    1. Find the index of the last HumanMessage to isolate the current turn.
    2. Collect all AI messages generated after that human message.
    3. Exclude the supervisor's optional greeting/routing message from the count.
    4. If ≤ 1 agent ran, return {} (no synthesis needed — the agent's message stands).
    5. If multiple agents ran, format their responses and invoke the LLM to merge them.
    6. Append the synthesized AIMessage to the state as the final visible response.

    Args:
        state: The global AgentState containing the full message history.
        config: LangChain execution configuration.

    Returns:
        A partial state dict with the synthesized `messages` list,
        or an empty dict if synthesis was skipped or failed.
    """
    messages = state.get("messages", [])

    # ── 1. Find the boundary of the current conversation turn ────────────────
    # We only want to synthesize messages that were generated THIS turn, not
    # messages from previous turns which are already user-visible history.
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if hasattr(messages[i], "type") and messages[i].type == "human":
            last_human_idx = i
            break

    if last_human_idx == -1:
        # No human message found — nothing to synthesize
        return {}

    # ── 2. Collect all AI messages generated in the current turn ──────────────
    current_turn_messages = messages[last_human_idx + 1 :]
    filtered_turn_messages = filter_tool_messages(current_turn_messages)

    new_ai_messages = filter_ai_messages(filtered_turn_messages)

    # ── 3. Build synthesis prompt and call the LLM ────────────────────────────
    has_image_context = (
        bool(state.get("image_base64")) or state.get("image_is_safe") is False
    )

    try:
        llm = get_llm(
            temperature=SYNTHESIZER_LLM_CONFIG.temperature,
            cache=False if has_image_context else SYNTHESIZER_LLM_CONFIG.default_cache,
        )

        prompt_messages = [
            get_synthesizer_system_message(),
            get_agent_responses_message(new_ai_messages),
        ]

        response = await llm.ainvoke(prompt_messages, config=config)

        # ── 6. Return the merged response as the final visible AI message ─────
        # The UI picks up the LAST AI message, so this becomes what the user sees.
        return {"messages": [AIMessage(content=response.content)]}

    except Exception as e:
        logger.exception("Synthesizer failed: %s", e)
        # Return {} — the individual agent responses will be shown instead
        return {}
