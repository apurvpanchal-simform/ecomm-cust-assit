import os
from typing import Any

from dotenv import load_dotenv
from langsmith import traceable
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from app.tools.faq_search import search_faq
from app.graph.state import AgentState
from app.schemas.agent import AgentResponse
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
"""

@traceable(
    name="faq_node",
    metadata={
        "agent": "faq",
    },
)
async def faq_node(state: AgentState, config: RunnableConfig) -> dict:

    query = state.get("query", "")
    # Only keep the last 8 messages for context to prevent bloated
    # conversations on resumed threads (full history stays in checkpointer).
    # The query is already in messages as a HumanMessage (added by the /chat endpoint).
    all_messages = list(state.get("messages", []))
    recent_messages = all_messages[-8:] if len(all_messages) > 8 else all_messages
    
    # Pre-fetch context using the search tool directly in Python
    try:
        context = str(await search_faq.ainvoke({"query": query})).strip()
    except Exception as e:
        context = f"Error performing search: {e}"

    context_message = SystemMessage(
        content=(
            f"Here is the context retrieved from the FAQ knowledge base for the user's query:\n"
            f"[[CONTEXT START]]\n"
            f"{context}\n"
            f"[[CONTEXT END]]\n\n"
            f"Answer the user's query using ONLY the provided context. Follow all guidelines.\n"
            f"IMPORTANT: If previous messages in the conversation history have already answered parts of the user's query (such as order details), do NOT repeat or comment on those parts. Focus strictly on answering the general FAQ question."
        )
    )

    conversation = [SystemMessage(content=SYSTEM_PROMPT)]
    
    chat_summary = state.get("chat_summary", "")
    if chat_summary:
        conversation.append(SystemMessage(content=f"Summary of earlier conversation:\n{chat_summary}"))
        
    conversation.append(context_message)
    conversation += recent_messages

    llm = get_llm(temperature=0.4)

    try:
        response = await llm.ainvoke(conversation, config=config)
        resolution_text = str(response.content).strip()
        
        last_msg = all_messages[-1] if all_messages else None
        if last_msg and hasattr(last_msg, 'type') and last_msg.type == "ai":
            merged_content = f"{last_msg.content}\n\n{resolution_text}"
            new_messages = [AIMessage(content=merged_content, id=last_msg.id)]
            final_resolution_text = merged_content
        else:
            new_messages = [AIMessage(content=resolution_text)]
            final_resolution_text = resolution_text
            
        error = None

    except Exception as exc:
        resolution_text = "Something went wrong while processing your request."
        last_msg = all_messages[-1] if all_messages else None
        if last_msg and hasattr(last_msg, 'type') and last_msg.type == "ai":
            merged_content = f"{last_msg.content}\n\n{resolution_text}"
            new_messages = [AIMessage(content=merged_content, id=last_msg.id)]
            final_resolution_text = merged_content
        else:
            new_messages = [AIMessage(content=resolution_text)]
            final_resolution_text = resolution_text
            
        error = str(exc)

    agent_response = AgentResponse(
        resolution_text=final_resolution_text,
    )

    return {
        "messages": new_messages,
        "agent_response": agent_response.model_dump(mode="json"),
        "error": error,
        "executed_agents": state.get("executed_agents", []) + ["faq"],
    }
