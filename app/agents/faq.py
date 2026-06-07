import os
from typing import Any

from dotenv import load_dotenv
from langsmith import traceable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from app.tools.faq_search import search_faq
from app.graph.state import AgentState
from app.schemas.agent import AgentResponse, TicketCategory

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


@traceable(name="faq_generation")
def generate_faq_response(
    agent,
    conversation: list,
    config: RunnableConfig,
) -> tuple[str, list, str | None]:

    new_messages = []

    max_iters = int(os.getenv("MAX_ITERATIONS", "6"))
    for _ in range(max_iters):
        response = agent.invoke(conversation, config=config)
        conversation.append(response)
        new_messages.append(response)

        if not response.tool_calls:
            break

        tool_msgs = []
        for tc in response.tool_calls:
            try:
                # `search_faq` takes `query` as argument
                result = (
                    str(_TOOL_MAP[tc["name"]].invoke(tc["args"])).strip()
                    or "No result returned."
                )
            except Exception as e:
                result = f"Error: {e}"

            tool_msgs.append(
                ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"])
            )

        conversation.extend(tool_msgs)
        new_messages.extend(tool_msgs)

    else:
        msg = "I'm having trouble processing your question. Please try again."
        new_messages.append(AIMessage(content=msg))
        return msg, new_messages, "max_iterations_exceeded"

    final_ai = next(
        (
            m
            for m in reversed(new_messages)
            if isinstance(m, AIMessage) and not m.tool_calls
        ),
        None,
    )

    if final_ai and isinstance(final_ai.content, list):
        text = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in final_ai.content
        )
    else:
        text = str(final_ai.content) if final_ai else "No response generated."

    return text, new_messages, None


@traceable(
    name="faq_node",
    metadata={
        "agent": "faq",
    },
)
async def faq_node(state: AgentState, config: RunnableConfig) -> dict:

    query = state.get("query", "")
    # Only keep the last 12 messages for context to prevent bloated
    # conversations on resumed threads (full history stays in checkpointer).
    # The query is already in messages as a HumanMessage (added by the /chat endpoint).
    all_messages = list(state.get("messages", []))
    recent_messages = all_messages[-12:] if len(all_messages) > 12 else all_messages
    
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

    from app.services.llm import get_llm
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
        requires_human = False
        confidence = 1.0

        if "ToolNotFoundResponse" in context or "No relevant FAQ articles" in context or "don't have a confident answer" in context:
            requires_human = True
            confidence = 0.0

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
        requires_human = True
        confidence = 0.0

    agent_response = AgentResponse(
        resolution_text=final_resolution_text,
        confidence_score=confidence,
        ticket_category=TicketCategory.FAQ,
        requires_human=requires_human,
        escalation_reason=error,
    )

    return {
        "messages": new_messages,
        "agent_response": agent_response.model_dump(mode="json"),
        "structured_output": agent_response.model_dump(mode="json"),
        "error": error,
        "executed_agents": state.get("executed_agents", []) + ["faq"],
    }
