import os
from typing import Any

from dotenv import load_dotenv
from langsmith import traceable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from app.tools.faq_search import search_faq
from app.graph.state import AgentState
from app.schemas.response import AgentResponse, TicketCategory

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
"""

@traceable(name="faq_generation")
def generate_faq_response(
    agent,
    conversation: list,
    config: RunnableConfig,
) -> tuple[str, list, str | None]:
    
    new_messages = []
    
    for _ in range(6): # MAX_ITERATIONS
        response = agent.invoke(conversation, config=config)
        conversation.append(response)
        new_messages.append(response)

        if not response.tool_calls:
            break

        tool_msgs = []
        for tc in response.tool_calls:
            try:
                # `search_faq` takes `query` as argument
                result = str(_TOOL_MAP[tc["name"]].invoke(tc["args"])).strip() or "No result returned."
            except Exception as e:
                result = f"Error: {e}"
            
            tool_msgs.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))

        conversation.extend(tool_msgs)
        new_messages.extend(tool_msgs)
        
    else:
        msg = "I'm having trouble processing your question. Please try again."
        new_messages.append(AIMessage(content=msg))
        return msg, new_messages, "max_iterations_exceeded"

    final_ai = next((m for m in reversed(new_messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    
    if final_ai and isinstance(final_ai.content, list):
        text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in final_ai.content)
    else:
        text = str(final_ai.content) if final_ai else "No response generated."

    return text, new_messages, None


@traceable(
    name="faq_node",
    metadata={
        "agent": "faq",
    },
)
def faq_node(state: AgentState, config: RunnableConfig) -> dict:
    
    query = state.get("query", "")
    conversation = [SystemMessage(content=SYSTEM_PROMPT)] + list(state.get("messages", []))
    
    if query:
        msg = HumanMessage(content=query)
        conversation.append(msg)

    model_name = os.getenv("PRIMARY_MODEL")
    if not model_name:
        raise RuntimeError("PRIMARY_MODEL environment variable is not set.")
        
    llm = ChatGroq(model=model_name)
    agent = llm.bind_tools(_FAQ_TOOLS)
    
    try:
        resolution_text, new_messages, error = generate_faq_response(
            agent,
            conversation,
            config,
        )
        requires_human = error is not None
        confidence = 0.0 if error else 1.0

    except Exception as exc:
        resolution_text = "Something went wrong while processing your request."
        new_messages = [AIMessage(content=resolution_text)]
        error = str(exc)
        requires_human = True
        confidence = 0.0

    agent_response = AgentResponse(
        resolution_text=resolution_text,
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
    }