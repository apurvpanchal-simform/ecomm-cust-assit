import os
import re
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from app.graph.state import AgentState
from app.schemas.response import AgentResponse, TicketCategory
from app.services.order_service import LocalOrderService

load_dotenv()

ORDER_PATTERN = re.compile(r"\bord[-\s]?(\d+)\b", re.IGNORECASE)

SYSTEM_PROMPT = """\
You are a customer support agent specializing in order management.

Use ONLY the real order data provided. Never invent order details.
If the order is not found, explain this clearly.

ORDER DATA:
{order_data}
"""

def order_node(state: AgentState) -> dict:
    message = state.get("query", "")
    customer_id = state.get("customer_id", "cust-001")

    # Extract order id from query
    match = ORDER_PATTERN.search(message)
    order_id = f"ord-{match.group(1)}" if match else None

    # Load orders
    service = LocalOrderService()
    orders = service.get_order_history(customer_id)

    # Find matching order
    order = (
        next((o for o in orders if o.get("id") == order_id), None)
        if order_id
        else (orders[0] if orders else None)
    )

    if not order:
        order_data = "No matching order found for this customer."
    else:
        order_data = json.dumps(order, indent=2)

    print(order_data)

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("PRIMARY_MODEL")
    )

    structured_llm = llm.with_structured_output(
        AgentResponse
    )

    response = structured_llm.invoke([
        SystemMessage(
            content=SYSTEM_PROMPT.format(
                order_data=order_data
            )
        ),
        HumanMessage(content=message),
    ])

    return {
        "support_response": AgentResponse(
            resolution_text=response.resolution_text,
            confidence_score=response.confidence_score,
            ticket_category=TicketCategory.ORDER,
            requires_human=response.requires_human,
            sentiment_score=response.sentiment_score,
            suggested_actions=response.suggested_actions,
            escalation_reason=response.escalation_reason,
            order_id=response.order_id,
        )
    }