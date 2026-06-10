import os
from typing import Any

from dotenv import load_dotenv
from langsmith import traceable
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from app.tools.faq_search import search_faq
from app.graph.state import AgentState
from app.graph.utils import filter_tool_messages
from app.services.llm import get_llm

load_dotenv()

_FAQ_TOOLS = [search_faq]
_TOOL_MAP: dict[str, Any] = {t.name: t for t in _FAQ_TOOLS}

SYSTEM_PROMPT = """You are a customer support agent for an e-commerce platform.
Help customers with general questions about policies, shipping, returns, and company operations.

Rules:
1. You MUST use the `search_faq` tool to retrieve context before answering.
2. Answer ONLY using the provided context from the tool.
3. Do not invent policies or information.
4. If the answer is not in the context, clearly state that and suggest contacting human support.
5. Be professional, concise, and helpful.
6. Summarize the tool results in natural, friendly language.
7. If the exact answer or data you need is already present in the 'Summary of earlier conversation', you may use it directly without making a duplicate tool call.
8. CRITICAL: If the `search_faq` tool returns an "Error:" or fails, DO NOT call the tool again. Immediately apologize to the user and explain that the search service is temporarily unavailable.
9. CRITICAL: If you receive a "specific task for this turn", you MUST prioritize that task and ignore unrelated parts of the user's broader conversation.
"""

@traceable(
    name="faq_node",
    metadata={
        "agent": "faq",
    },
)
async def faq_node(state: AgentState, config: RunnableConfig) -> dict:

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
        conversation.append(SystemMessage(content=f"Summary of earlier conversation:\n{chat_summary}"))
        
    if sub_query:
        conversation.append(SystemMessage(content=f"Your specific task for this turn: {sub_query}"))
        
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
