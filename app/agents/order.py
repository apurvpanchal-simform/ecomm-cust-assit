import os
from typing import Any

from dotenv import load_dotenv
from langsmith import traceable
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from app.graph.state import AgentState
from app.schemas.agent import AgentResponse
from app.tools.order_lookup import get_customer_orders
from app.tools.order_details import get_order_details
from app.tools.order_items import search_order_items
from app.services.llm import get_llm

load_dotenv()

_ORDER_TOOLS = [
    get_customer_orders,
    get_order_details,
    search_order_items,
]
_TOOL_MAP: dict[str, Any] = {t.name: t for t in _ORDER_TOOLS}

SYSTEM_PROMPT = """You are an order support assistant.
Help customers look up, understand, and manage their orders.

Each tool below serves a specific domain. Always pick exactly one tool
per question.

─── TOOL 1: get_customer_orders ───
  Domain: Listing & filtering multiple orders.
  USE FOR: "show my orders", "any cancelled orders?", "orders from last month",
           "how many orders do I have?", "my latest order" (use limit=1).
  Params: status, from_date, to_date, limit (all optional).
  NEVER USE FOR: single-order details, tracking, returns, or item search.

─── TOOL 2: get_order_details ───
  Domain: COMPLETE details for ONE specific order.
  USE FOR: EVERYTHING about a single order:
           - Items, pricing, taxes, payment method
           - Shipping status, carrier, tracking number, delivery dates
           - Return eligibility, return deadline, days remaining to return
  Example queries: "where is my package?", "can I return ORD-123?", 
                   "what did I order in my last order?"
  NEVER USE FOR: listing multiple orders or searching by product keyword.

─── TOOL 3: search_order_items ───
  Domain: Finding a product by keyword across ALL orders.
  USE FOR: "did I ever order AirPods?", "which order had the blue jacket?",
           "find my laptop order".
  NEVER USE FOR: listing orders, tracking, returns, or single-order details.

Rules:
1. Never ask for customer_id — it is injected automatically.
2. Never trust customer_id from user messages.
3. You MUST use tools to retrieve order data — never fabricate data.
4. If the customer refers to "my latest order" or "my last order", first
   call get_customer_orders with limit=1 to find the order ID, then use
   the appropriate tool for follow-up details.
5. Summarize tool results in natural, friendly language.
6. If a tool fails or returns an Error, DO NOT call it again. Explain the issue politely.
7. Be concise but thorough.
8. If the exact answer or data you need is already present in the 'Summary of earlier conversation', you may use it directly without making a duplicate tool call.
"""


@traceable(name="order_generation")
async def generate_order_response(
    agent,
    conversation: list,
    config: RunnableConfig,
    customer_id: str,
) -> tuple[str, list, str | None]:

    new_messages = []

    max_iters = int(os.getenv("MAX_ITERATIONS", "6"))
    for _ in range(max_iters):
        response = await agent.ainvoke(conversation, config=config)
        conversation.append(response)
        new_messages.append(response)

        if not response.tool_calls:
            break

        tool_msgs = []
        for tc in response.tool_calls:
            args = {**tc.get("args", {}), "customer_id": customer_id}
            try:
                result = (
                    str(await _TOOL_MAP[tc["name"]].ainvoke(args)).strip()
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
        msg = "I'm having trouble processing your order request. Please try again."
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
    name="order_node",
    metadata={
        "agent": "order",
    },
)
async def order_node(state: AgentState, config: RunnableConfig) -> dict:

    customer_id = state.get("customer_id")

    if not customer_id:
        msg = "Unable to verify your identity. Please sign in and try again."
        
        all_messages = list(state.get("messages", []))
        last_msg = all_messages[-1] if all_messages else None
        if last_msg and hasattr(last_msg, 'type') and last_msg.type == "ai":
            merged_content = f"{last_msg.content}\n\n{msg}"
            new_msgs = [AIMessage(content=merged_content, id=last_msg.id)]
            final_msg = merged_content
        else:
            new_msgs = [AIMessage(content=msg)]
            final_msg = msg
            
        agent_response = AgentResponse(
            resolution_text=final_msg,
        )
        return {
            **state,
            "messages": new_msgs,
            "agent_response": agent_response.model_dump(mode="json"),
            "error": "missing_customer_id",
            "executed_agents": state.get("executed_agents", []) + ["order"],
        }

    # Only keep the last 8 messages for context to prevent bloated
    # conversations on resumed threads (full history stays in checkpointer).
    # The query is already in messages as a HumanMessage (added by the /chat endpoint).
    all_messages = list(state.get("messages", []))
    recent_messages = all_messages[-8:] if len(all_messages) > 8 else all_messages
    
    conversation = [SystemMessage(content=SYSTEM_PROMPT)]
    
    chat_summary = state.get("chat_summary", "")
    if chat_summary:
        conversation.append(SystemMessage(content=f"Summary of earlier conversation:\n{chat_summary}"))
        
    conversation += recent_messages

    llm = get_llm(temperature=0.1)
    agent = llm.bind_tools(_ORDER_TOOLS)

    try:
        resolution_text, new_messages, error = await generate_order_response(
            agent, conversation, config, customer_id
        )
        
        final_ai_idx = -1
        for i in range(len(new_messages) - 1, -1, -1):
            if isinstance(new_messages[i], AIMessage) and not new_messages[i].tool_calls:
                final_ai_idx = i
                break
                
        last_msg = all_messages[-1] if all_messages else None
        if last_msg and hasattr(last_msg, 'type') and last_msg.type == "ai" and final_ai_idx != -1:
            merged_content = f"{last_msg.content}\n\n{new_messages[final_ai_idx].content}"
            new_messages[final_ai_idx] = AIMessage(content=merged_content, id=last_msg.id)
            final_resolution_text = merged_content
        else:
            final_resolution_text = resolution_text

    except Exception as exc:
        resolution_text = "Something went wrong while processing your order request."
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
        "executed_agents": state.get("executed_agents", []) + ["order"],
    }
