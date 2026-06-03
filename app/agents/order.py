from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app.graph.state import AgentState
from app.schemas.response import AgentResponse, TicketCategory
from app.tools.order_lookup import get_customer_orders
from app.tools.order_details import get_order_details
from app.tools.order_items import search_order_items

load_dotenv()

_ORDER_TOOLS = [
    get_customer_orders,
    get_order_details,
    search_order_items,
]
_TOOL_MAP: dict[str, Any] = {t.name: t for t in _ORDER_TOOLS}

_SYSTEM_PROMPT = """You are an order support assistant.
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
6. If a tool fails, explain the issue politely.
7. Be concise but thorough.
"""

def _create_order_agent() -> Any:
    model_name = os.getenv("PRIMARY_MODEL")
    if not model_name:
        raise RuntimeError("PRIMARY_MODEL environment variable is not set.")
    return ChatGroq(model=model_name).bind_tools(_ORDER_TOOLS)

def _build_response(
    state: AgentState,
    new_messages: list,
    text: str,
    confidence: float = 1.0,
    requires_human: bool = False,
    error: str | None = None,
) -> AgentState:
    resp = AgentResponse(
        resolution_text=text,
        confidence_score=confidence,
        ticket_category=TicketCategory.ORDER,
        requires_human=requires_human,
        escalation_reason=error,
    )
    return {
        **state,
        "messages": new_messages,
        "agent_response": resp,
        "structured_output": resp.model_dump(mode="json"),
        "error": error,
    }

def order_node(state: AgentState, config: RunnableConfig) -> AgentState:
    customer_id = state.get("customer_id")
    if not customer_id:
        msg = "Unable to verify your identity. Please sign in and try again."
        return _build_response(state, [AIMessage(content=msg)], msg, error="missing_customer_id")

    query = state.get("query", "")
    conversation = [SystemMessage(content=_SYSTEM_PROMPT)] + list(state.get("messages", []))
    new_messages = []

    if query:
        msg = HumanMessage(content=query)
        conversation.append(msg)
        new_messages.append(msg)

    try:
        agent = _create_order_agent()
        
        for _ in range(6): # MAX_ITERATIONS
            response = agent.invoke(conversation, config=config)
            conversation.append(response)
            new_messages.append(response)

            if not response.tool_calls:
                break

            tool_msgs = []
            for tc in response.tool_calls:
                args = {**tc.get("args", {}), "customer_id": customer_id}
                try:
                    result = str(_TOOL_MAP[tc["name"]].invoke(args)).strip() or "No result returned."
                except Exception as e:
                    result = f"Error: {e}"
                
                tool_msgs.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))

            conversation.extend(tool_msgs)
            new_messages.extend(tool_msgs)
            
        else:
            msg = "I'm having trouble processing your order request. Please try again."
            new_messages.append(AIMessage(content=msg))
            return _build_response(state, new_messages, msg, 0.0, True, "max_iterations_exceeded")

        final_ai = next((m for m in reversed(new_messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
        
        if final_ai and isinstance(final_ai.content, list):
            text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in final_ai.content)
        else:
            text = str(final_ai.content) if final_ai else "No response generated."

        return _build_response(state, new_messages, text)

    except Exception as exc:
        msg = "Something went wrong while processing your order request."
        new_messages.append(AIMessage(content=msg))
        return _build_response(state, new_messages, msg, 0.0, True, str(exc))