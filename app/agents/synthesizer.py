"""
Agent node that synthesizes outputs from multiple sub-agents into a single, cohesive user response.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import observe

from app.graph.state import AgentState
from app.graph.utils import filter_tool_messages
from app.services.llm_factory import get_llm

logger = logging.getLogger(__name__)

SYNTHESIZER_PROMPT = """You are an e-commerce assistant. You receive raw responses from multiple backend agents that ran during a single conversation turn.

## Task
Merge all agent responses into one cohesive, natural reply for the user.

## Rules
1. Combine information naturally—don't mechanically list each agent's output.
2. If one response is a refusal (e.g., "I can't help with coding"), weave it in gracefully alongside valid information.
3. Never invent facts—use only what the agents provided.
4. Keep the tone warm and helpful.
5. NEVER use inline code (backticks `) to format labels or monetary amounts (e.g., do NOT write `Subtotal: \`$10\`` or `\`**Tax**\``). Use standard bold text instead.
"""


@observe(name="synthesizer_node")
async def synthesizer_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Synthesizes multiple agent responses into a single, cohesive response.

    This node acts as the final step in a multi-agent conversation turn before summarization.
    It evaluates the AI messages generated in the current turn. If multiple sub-agents
    (e.g., FAQ and Order) were triggered by the supervisor, this node takes their disparate
    responses and uses an LLM to merge them into a single, natural, and helpful reply for the user.

    Flow:
    1. Identifies the last human message in the state to isolate the current conversation turn.
    2. Collects all AI messages generated after that human message, filtering out tool calls.
    3. Excludes the supervisor's routing/greeting message from the agent count to accurately
       determine if multiple sub-agents ran.
    4. If 1 or 0 sub-agents ran, it skips synthesis and returns an empty dict (allowing the
       single agent's message to be presented directly to the user).
    5. If multiple sub-agents ran, it constructs a prompt containing all their raw responses
       and invokes the LLM to generate a unified response.
    6. Appends the newly synthesized message to the state, tagged with `name="synthesizer"`.

    Args:
        state (AgentState): The global state of the LangGraph containing the conversation history.
        config (RunnableConfig): Configuration parameters for the LangChain execution.

    Returns:
        dict: A dictionary containing the newly synthesized `messages` to append to the state,
              or an empty dictionary if synthesis is skipped.
    """
    messages = state.get("messages", [])

    # Find the last human message
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if hasattr(messages[i], "type") and messages[i].type == "human":
            last_human_idx = i
            break

    if last_human_idx == -1:
        return {}

    current_turn_messages = messages[last_human_idx + 1 :]
    filtered_turn_messages = filter_tool_messages(current_turn_messages)

    # Get all AI messages generated AFTER the last human message
    new_ai_messages = [
        m for m in filtered_turn_messages if hasattr(m, "type") and m.type == "ai"
    ]

    # Exclude supervisor messages from the count so that 1 agent + 1 supervisor = 1 agent response
    agent_responses = [
        m for m in new_ai_messages if getattr(m, "name", "") != "supervisor"
    ]

    # Only synthesize if there are multiple agent responses
    if len(agent_responses) <= 1:
        return {}

    try:
        llm = get_llm(temperature=0.3)

        # Format the agent responses for the LLM
        agent_responses_text = "Raw Agent Responses to combine:\n"
        for i, msg in enumerate(new_ai_messages):
            source = getattr(msg, "name", f"Agent_{i + 1}")
            agent_responses_text += f"- [{source}]: {msg.content}\n"

        prompt_messages = [
            SystemMessage(content=SYNTHESIZER_PROMPT),
            HumanMessage(content=agent_responses_text),
        ]

        response = await llm.ainvoke(prompt_messages, config=config)

        # Append the final synthesized message.
        # The UI will pick this up as the LAST AI message!
        return {"messages": [AIMessage(content=response.content, name="synthesizer")]}
    except Exception as e:
        logger.exception(f"Synthesizer failed: {e}")
        return {}
