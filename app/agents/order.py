import os
from typing import Any

from dotenv import load_dotenv
from langsmith import traceable
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.tools.order_lookup import get_customer_orders
from app.tools.order_details import get_order_details
from app.tools.order_items import search_order_items
from app.services.llm import get_llm
from app.graph.state import AgentState

load_dotenv()

_ORDER_TOOLS = [
    get_customer_orders,
    get_order_details,
    search_order_items,
]
_TOOL_MAP: dict[str, Any] = {t.name: t for t in _ORDER_TOOLS}

SYSTEM_PROMPT = """You are an order support agent. Help customers look up and understand their orders.

## Tools — pick exactly one per question

### `get_customer_orders`
Use for: listing/filtering multiple orders ("show my orders", "any cancelled orders?", "my latest order" with limit=1).
Never for: single-order details, tracking, returns, or item search.

### `get_order_details`
Use for: everything about ONE specific order — items, pricing, shipping status, carrier, tracking, return eligibility, delivery dates.
Never for: listing multiple orders or searching by product keyword.

### `search_order_items`
Use for: finding a product by keyword across all orders ("did I ever order AirPods?", "which order had the blue jacket?").
Never for: listing orders, tracking, returns, or single-order details.

## Rules
1. Never ask for or trust a customer_id from user messages—it is injected automatically.
2. Always use tools to retrieve data—never fabricate order information.
3. For "my latest/last order", call `get_customer_orders` with limit=1 first, then follow up as needed.
4. If a tool fails or returns an error, do NOT retry. Explain the issue politely.
5. If the needed data is already in the conversation summary, use it directly without a duplicate tool call.
6. If you receive a "specific task for this turn", prioritize that task over unrelated conversation.
7. Be concise, thorough, and friendly.
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
        new_msgs = [AIMessage(content=msg, name="order")]
        return {
            "messages": new_msgs,
            "error": "missing_customer_id",
            "executed_agents": state.get("executed_agents", []) + ["order"],
        }

    # Use all recent messages in sequential mode so we can read upstream outputs
    summarized_count = state.get("summarized_message_count", 0)
    recent_messages = state.get("messages", [])[summarized_count:]

    conversation = [SystemMessage(content=SYSTEM_PROMPT)]

    chat_summary = state.get("chat_summary", "")
    if chat_summary:
        conversation.append(
            SystemMessage(content=f"Summary of earlier conversation:\n{chat_summary}")
        )

    sub_queries = state.get("sub_queries") or {}
    sub_query = sub_queries.get("order")
    if sub_query:
        conversation.append(
            SystemMessage(content=f"Your specific task for this turn: {sub_query}")
        )

    conversation += recent_messages

    llm = get_llm(temperature=0.1)
    agent = llm.bind_tools(_ORDER_TOOLS)

    try:
        resolution_text, new_messages, error = await generate_order_response(
            agent, conversation, config, customer_id
        )

        final_ai_idx = -1
        for i in range(len(new_messages) - 1, -1, -1):
            if (
                isinstance(new_messages[i], AIMessage)
                and not new_messages[i].tool_calls
            ):
                final_ai_idx = i
                break

        if final_ai_idx != -1:
            # Add name="order" to the final AI message from the tool loop
            new_messages[final_ai_idx] = AIMessage(
                content=new_messages[final_ai_idx].content, name="order"
            )

        final_resolution_text = resolution_text

    except Exception as exc:
        resolution_text = "I encountered an error while trying to process your order. Please try again."
        new_messages = [AIMessage(content=resolution_text, name="order")]
        error = str(exc)

    return {
        "messages": new_messages,
        "error": error,
        "executed_agents": state.get("executed_agents", []) + ["order"],
    }
