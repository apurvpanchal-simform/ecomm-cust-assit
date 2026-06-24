"""
Customer conversation history and management routes.
"""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request

from app.middleware.rate_limit import rate_limit_customer
from app.schemas.api import ConversationItem

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/chat/conversations", response_model=List[ConversationItem])
async def list_conversations(
    request: Request,
    customer_id: str = Depends(rate_limit_customer),
):
    """
    Retrieve all past conversation threads for the authenticated customer.

    Args:
        request: FastAPI request (provides pool access).
        customer_id: Authenticated customer ID from JWT.

    Returns:
        List of ConversationItem schemas sorted by most-recently-updated.
    """
    async with request.app.state.pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT conversation_id, title, updated_at
            FROM customer_conversations
            WHERE customer_id = %s
            ORDER BY updated_at DESC
            """,
            (customer_id,),
        )
        result = await cursor.fetchall()
    return [ConversationItem(**row) for row in result]


def _flush_ai_messages(
    current_turn_ai_messages: list, formatted_messages: list
) -> None:
    """
    Flush accumulated AI messages for the current turn into `formatted_messages`.

    Prefers the synthesizer's message over other AI messages so the user sees
    the final merged response rather than an intermediate sub-agent response.
    Modifies `current_turn_ai_messages` in place (clears it after flushing).

    Args:
        current_turn_ai_messages: AI messages accumulated so far this turn.
        formatted_messages: Output list to append the final formatted message to.
    """
    if not current_turn_ai_messages:
        return

    # Prefer the synthesizer's merged response over individual agent responses
    synth_msg = next(
        (
            m
            for m in current_turn_ai_messages
            if getattr(m, "name", "") == "synthesizer"
        ),
        None,
    )
    final_msg = synth_msg if synth_msg else current_turn_ai_messages[-1]
    
    content_str = ""
    if final_msg.content:
        if isinstance(final_msg.content, list):
            text_parts = [
                (
                    item.get("text", "")
                    if isinstance(item, dict) and item.get("type") == "text"
                    else str(item) if isinstance(item, str) else ""
                )
                for item in final_msg.content
            ]
            content_str = " ".join(text_parts).strip()
        else:
            content_str = str(final_msg.content)

    formatted_messages.append({"role": "assistant", "content": content_str})
    current_turn_ai_messages.clear()


@router.get("/chat/history/{conversation_id}")
async def get_chat_history(
    conversation_id: str,
    request: Request,
    customer_id: str = Depends(rate_limit_customer),
):
    """
    Retrieve formatted message history for a specific conversation.

    Validates ownership, reads the LangGraph state, and formats the raw
    LangChain messages into user/assistant role-based entries.  Multi-agent
    turns are collapsed so only the synthesizer (or last) AI message is shown.

    Args:
        conversation_id: The conversation thread ID.
        request: FastAPI request.
        customer_id: Authenticated customer ID from JWT.

    Returns:
        Dict with `status` and `messages` (list of {role, content, [images]}).

    Raises:
        HTTPException 403: If the conversation doesn't exist or isn't owned by this customer.
    """
    # ── 1. Validate ownership ─────────────────────────────────────────────────
    async with request.app.state.pool.connection() as conn:
        # First: does the conversation exist at all?
        cursor = await conn.execute(
            "SELECT customer_id, status FROM customer_conversations WHERE conversation_id = %s",
            (conversation_id,),
        )
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Second: does it belong to this customer?
    if row["customer_id"] != customer_id:
        raise HTTPException(status_code=403, detail="Access denied")


    # ── 2. Read graph state ───────────────────────────────────────────────────
    config = {"configurable": {"thread_id": conversation_id}}
    state = await request.app.state.graph.aget_state(config)
    messages = state.values.get("messages", [])

    # ── 3. Format messages into user/assistant roles ──────────────────────────
    formatted_messages = []
    current_turn_ai_messages = []

    for msg in messages:
        if msg.type == "human":
            # New human message — flush the previous turn's AI messages first
            _flush_ai_messages(current_turn_ai_messages, formatted_messages)
            if msg.content:
                # Extract text and images from potentially multimodal content
                text_content = ""
                images = []
                if isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_content += item.get("text", "")
                            elif item.get("type") == "image_url":
                                images.append(item.get("image_url", {}).get("url", ""))
                else:
                    text_content = str(msg.content)

                formatted_messages.append(
                    {"role": "user", "content": text_content, "images": images}
                )
        elif msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
            current_turn_ai_messages.append(msg)

    # Flush any remaining AI messages from the last turn
    _flush_ai_messages(current_turn_ai_messages, formatted_messages)

    return {"status": row["status"], "messages": formatted_messages}


@router.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    request: Request,
    customer_id: str = Depends(rate_limit_customer),
):
    """
    Delete a specific conversation thread from the database.

    Args:
        conversation_id: The conversation thread ID to delete.
        request: FastAPI request.
        customer_id: Authenticated customer ID from JWT.

    Returns:
        {"status": "deleted"} on success.

    Raises:
        HTTPException 404: If the conversation is not found or not owned by this customer.
    """
    async with request.app.state.pool.connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM customer_conversations WHERE conversation_id = %s AND customer_id = %s RETURNING conversation_id",
            (conversation_id, customer_id),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=404, detail="Conversation not found or not owned by user"
            )
        await conn.commit()
    return {"status": "deleted"}


@router.post("/chat/conversations/{conversation_id}/typing/customer")
async def customer_typing(
    conversation_id: str,
    request: Request,
    customer_id: str = Depends(rate_limit_customer),
):
    """
    Publish a customer typing event to the Redis support:events channel.
    """
    import json

    # 1. Validate ownership
    async with request.app.state.pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT customer_id FROM customer_conversations WHERE conversation_id = %s",
            (conversation_id,),
        )
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if row["customer_id"] != customer_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # 2. Publish to Redis Pub/Sub support:events channel
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client:
        await redis_client.publish(
            "support:events",
            json.dumps({"event": "customer_typing", "conversation_id": conversation_id}),
        )
    return {"status": "ok"}

