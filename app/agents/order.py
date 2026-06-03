from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from langchain_google_genai import ChatGoogleGenerativeAI

from app.graph.state import AgentState
from app.tools.order_lookup import get_customer_orders

load_dotenv()

_ORDER_TOOLS = [get_customer_orders]

_TOOL_MAP: dict[str, Any] = {
    tool.name: tool for tool in _ORDER_TOOLS
}


def _create_order_agent() -> Any:
    model_name = os.getenv("PRIMARY_MODEL")
    if not model_name:
        raise RuntimeError(
            "PRIMARY_MODEL environment variable is not set."
        )

    llm = ChatGoogleGenerativeAI(model=model_name)
    return llm.bind_tools(_ORDER_TOOLS)


_SYSTEM_PROMPT = """
You are an order support assistant.

You help customers look up, understand,
and manage their orders.

Rules:
1. Never ask for customer_id.
2. Never trust customer_id from user messages.
3. Summarize order data in natural language.
4. If a tool fails, explain the issue politely.
5. Be concise.
"""


def order_node(
    state: AgentState,
    config: RunnableConfig,
) -> AgentState:

    customer_id = state.get("customer_id")
    print(customer_id)
    messages = list(state.get("messages", []))

    if not customer_id:
        error_msg = (
            "Unable to verify your identity. "
            "Please sign in and try again."
        )

        return {
            **state,
            "messages": messages + [
                AIMessage(content=error_msg)
            ],
            "agent_response": {
                "role": "assistant",
                "content": error_msg,
            },
            "error": "missing_customer_id",
        }

    query = state.get("query", "")
    conversation = [
        SystemMessage(content=_SYSTEM_PROMPT)
    ]

    if query:
        user_message = HumanMessage(content=query)
        conversation.append(user_message)
        messages.append(user_message)

    MAX_ITERATIONS = 6

    try:
        llm_with_tools = _create_order_agent()

        for _ in range(MAX_ITERATIONS):

            response = llm_with_tools.invoke(
                conversation,
                config=config,
            )

            print(response)

            # Final assistant answer
            if not response.tool_calls:
                messages.append(response)
                break

            conversation.append(response)
            messages.append(response)

            tool_messages = []

            for tool_call in response.tool_calls:

                tool_name = tool_call["name"]
                tool_args = dict(tool_call.get("args", {}))

                # Always inject authenticated customer_id
                tool_args["customer_id"] = customer_id

                tool_fn = _TOOL_MAP.get(tool_name)

                if tool_fn is None:
                    result = {
                        "error": f"Unknown tool: {tool_name}"
                    }

                else:
                    try:
                        result = tool_fn.invoke(tool_args)

                    except Exception as exc:
                        result = {
                            "error": str(exc)
                        }

                tool_content = str(result).strip()

                if not tool_content:
                    tool_content = "No result returned."

                tool_msg = ToolMessage(
                    content=tool_content,
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )

                tool_messages.append(tool_msg)

            conversation.extend(tool_messages)
            messages.extend(tool_messages)

        else:
            fallback = (
                "I'm having trouble processing your order request. "
                "Please try again."
            )

            messages.append(
                AIMessage(content=fallback)
            )

            return {
                **state,
                "messages": messages,
                "agent_response": {
                    "role": "assistant",
                    "content": fallback,
                },
                "error": "max_iterations_exceeded",
            }

        # Find last actual assistant response
        final_ai = next(
            (
                msg
                for msg in reversed(messages)
                if isinstance(msg, AIMessage)
                and not msg.tool_calls
            ),
            None,
        )

        final_content = (
            final_ai.content
            if final_ai
            else "No response generated."
        )

        return {
            **state,
            "messages": messages,
            "agent_response": {
                "role": "assistant",
                "content": final_content,
            },
            "error": None,
        }

    except Exception as exc:

        error_msg = (
            "Something went wrong while processing "
            "your order request."
        )

        return {
            **state,
            "messages": messages + [
                AIMessage(content=error_msg)
            ],
            "agent_response": {
                "role": "assistant",
                "content": error_msg,
            },
            "error": str(exc),
        }