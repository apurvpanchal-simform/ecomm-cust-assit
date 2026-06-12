import os
from typing import Any

from dotenv import load_dotenv
from langfuse import observe
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from app.tools.faq_search import search_faq
from app.graph.state import AgentState
from app.graph.utils import filter_tool_messages
from app.services.llm_factory import get_llm

load_dotenv()

_FAQ_TOOLS = [search_faq]
_TOOL_MAP: dict[str, Any] = {t.name: t for t in _FAQ_TOOLS}

SYSTEM_PROMPT = """You are a FAQ support agent for an e-commerce platform. You answer questions about policies, shipping, returns, and general company info.

## Rules
1. Always use the `search_faq` tool to retrieve context before answering.
2. Answer strictly from the retrieved context—never invent policies or facts.
3. If the answer isn't in the context, say so and suggest contacting human support.
4. If the context or data you need is already in the conversation summary, use it directly without a duplicate tool call.
5. If `search_faq` returns an error, do NOT retry. Apologize and explain the service is temporarily unavailable.
6. If you receive a "specific task for this turn", prioritize that task over unrelated conversation.
7. Be professional, concise, and friendly.
"""


@observe(name="faq_node")
async def faq_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Handles user queries related to general platform questions, policies, and FAQs.

    This node is part of the LangGraph multi-agent architecture and acts as the FAQ specialist. 
    When the supervisor delegates a query to this node, it automatically searches the FAQ 
    knowledge base using the provided sub-query or the original user query, and then formulates 
    a response using only the retrieved context.

    Flow:
    1. Extracts the relevant query (`sub_query` or `query`) from the state.
    2. Directly invokes the `search_faq` tool to retrieve relevant documentation.
    3. Injects the retrieved context directly into the system prompt to prevent hallucination.
    4. Constructs a conversation array consisting of the system prompt, chat summary, 
       task instructions, retrieved context, and recent conversation history.
    5. Invokes the LLM to generate an answer based purely on the context.
    6. Appends the AI response to the state and marks the agent as executed.

    Args:
        state (AgentState): The global state of the conversation, containing history, 
                            sub-queries, and execution tracking.
        config (RunnableConfig): Configuration parameters for LangChain execution.

    Returns:
        dict: A dictionary containing:
            - `messages`: A list containing the newly generated AIMessage.
            - `error`: An error string if an exception occurred, otherwise None.
            - `executed_agents`: The updated list of agents that have run in this turn.
    """

    # In Sequential List without isolation, we just read all recent messages
    summarized_count = state.get("summarized_message_count", 0)
    recent_messages = state.get("messages", [])[summarized_count:]

    sub_queries = state.get("sub_queries") or {}
    sub_query = sub_queries.get("faq")
    search_term = sub_query if sub_query else state.get("query", "")

    # Pre-fetch context using the search tool directly in Python
    try:
        context = str(await search_faq.ainvoke({"query": search_term})).strip()
    except Exception as e:
        context = f"Error performing search: {e}"

    context_message = SystemMessage(
        content=(
            f"Here is the context retrieved from the FAQ knowledge base for the user's query:\n"
            f"[[CONTEXT START]]\n"
            f"{context}\n"
            f"[[CONTEXT END]]\n\n"
            f"Answer the user's query using ONLY the provided context. Follow all guidelines."
        )
    )

    conversation = [SystemMessage(content=SYSTEM_PROMPT)]

    chat_summary = state.get("chat_summary", "")
    if chat_summary:
        conversation.append(
            SystemMessage(content=f"Summary of earlier conversation:\n{chat_summary}")
        )

    if sub_query:
        conversation.append(
            SystemMessage(content=f"Your specific task for this turn: {sub_query}")
        )

    conversation.append(context_message)
    conversation += filter_tool_messages(recent_messages)

    llm = get_llm(temperature=0.4)

    try:
        response = await llm.ainvoke(conversation, config=config)
        resolution_text = str(response.content).strip()
        new_messages = [AIMessage(content=resolution_text, name="faq")]
        error = None

    except Exception as exc:
        resolution_text = "I encountered an error while trying to process your request. Please try again."
        new_messages = [AIMessage(content=resolution_text, name="faq")]
        error = str(exc)

    return {
        "messages": new_messages,
        "error": error,
        "executed_agents": state.get("executed_agents", []) + ["faq"],
    }
