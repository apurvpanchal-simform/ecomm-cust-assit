from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from app.graph.state import AgentState
from langsmith import traceable
from app.services.llm import get_llm
import logging

logger = logging.getLogger(__name__)

SYNTHESIZER_PROMPT = """You are an e-commerce assistant. You receive raw responses from multiple backend agents that ran during a single conversation turn.

## Task
Merge all agent responses into one cohesive, natural reply for the user.

## Rules
1. Combine information naturally—don't mechanically list each agent's output.
2. If one response is a refusal (e.g., "I can't help with coding"), weave it in gracefully alongside valid information.
3. Never invent facts—use only what the agents provided.
4. Keep the tone warm and helpful.
"""


@traceable(name="synthesizer_node")
async def synthesizer_node(state: AgentState, config: RunnableConfig) -> dict:
    """Synthesizes multiple agent responses into a single, cohesive response."""
    messages = state.get("messages", [])

    # Find the last human message
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if hasattr(messages[i], "type") and messages[i].type == "human":
            last_human_idx = i
            break

    if last_human_idx == -1:
        return {}

    # Get all AI messages generated AFTER the last human message that don't have tool calls
    new_ai_messages = [
        m
        for m in messages[last_human_idx + 1 :]
        if hasattr(m, "type") and m.type == "ai" and not getattr(m, "tool_calls", None)
    ]

    # Only synthesize if there are multiple responses
    if len(new_ai_messages) <= 1:
        return {}

    try:
        llm = get_llm(temperature=0.3)

        # Format the agent responses for the LLM
        agent_responses_text = "Raw Agent Responses to combine:\n"
        for i, msg in enumerate(new_ai_messages):
            source = getattr(msg, "name", f"Agent_{i+1}")
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
