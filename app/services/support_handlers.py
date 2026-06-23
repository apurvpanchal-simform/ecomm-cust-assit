"""
Support agent action handlers for processing conversation updates.
"""

import json
import logging
from langchain_core.messages import AIMessage, SystemMessage
from app.utils.summary import move_active_to_resolved_in_summary

logger = logging.getLogger(__name__)


async def _handle_agent_reply(
    conversation_id: str, message: str, agent_name: str, app_state
) -> None:
    """Handle a support agent reply by updating the LangGraph checkpoint state and DB, and broadcasting to Redis."""
    graph = app_state.graph
    redis_client = app_state.redis
    pool = app_state.pool

    # 1. Write reply to LangGraph checkpoint history
    config = {"configurable": {"thread_id": conversation_id}}
    try:
        await graph.aupdate_state(
            config,
            {"messages": [AIMessage(content=message, name="support_agent")]},
        )
    except Exception as e:
        logger.error("Failed to save support reply in graph: %s", e)

    # 2. Update the conversation timestamp
    try:
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE customer_conversations SET updated_at = NOW() WHERE conversation_id = %s",
                (conversation_id,),
            )
            await conn.commit()
    except Exception as e:
        logger.error("Failed to update conversation timestamp in DB: %s", e)

    # 3. Publish to customer's WebSocket via Redis Pub/Sub
    notification = json.dumps(
        {
            "event": "agent_reply",
            "content": message,
            "agent_name": agent_name,
            "conversation_id": conversation_id,
        }
    )
    if redis_client:
        await redis_client.publish(f"chat:reply:{conversation_id}", notification)
        await redis_client.publish(
            "support:events",
            json.dumps({"event": "agent_reply", "conversation_id": conversation_id}),
        )
    logger.info("Support agent %s replied to %s via WS", agent_name, conversation_id)


async def _handle_agent_typing(
    conversation_id: str, agent_name: str, app_state
) -> None:
    """Handle a support agent typing indicator."""
    pool = getattr(app_state, "pool", None)
    if pool:
        try:
            async with pool.connection() as conn:
                await conn.execute(
                    "UPDATE customer_conversations SET updated_at = NOW() WHERE conversation_id = %s",
                    (conversation_id,),
                )
                await conn.commit()
        except Exception as e:
            logger.error("Failed to update timestamp during agent typing: %s", e)

    redis_client = getattr(app_state, "redis", None)
    if redis_client:
        notification = json.dumps(
            {
                "event": "agent_typing",
                "agent_name": agent_name,
                "conversation_id": conversation_id,
            }
        )
        await redis_client.publish(f"chat:reply:{conversation_id}", notification)


async def _handle_agent_join(conversation_id: str, agent_name: str, app_state) -> None:
    """Handle a support agent joining a conversation."""
    pool = app_state.pool
    redis_client = app_state.redis

    try:
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE customer_conversations SET assigned_agent = %s, updated_at = NOW() WHERE conversation_id = %s",
                (agent_name, conversation_id),
            )
            await conn.commit()
    except Exception as e:
        logger.error("Failed to assign agent in DB: %s", e)

    notification = json.dumps(
        {
            "event": "agent_joined",
            "content": f"🟢 **{agent_name} (Support Agent) has joined the chat.**",
            "agent_name": agent_name,
            "conversation_id": conversation_id,
        }
    )
    if redis_client:
        await redis_client.publish(f"chat:reply:{conversation_id}", notification)
        await redis_client.publish(
            "support:events",
            json.dumps({"event": "agent_joined", "conversation_id": conversation_id}),
        )
    logger.info("Agent %s joined conversation %s via WS", agent_name, conversation_id)


async def _handle_agent_resolve(conversation_id: str, app_state) -> None:
    """Handle a support agent resolving a conversation."""
    pool = app_state.pool
    redis_client = app_state.redis
    graph = app_state.graph

    # Pre-define so it is always bound even if DB ops fail
    updated_summary = ""

    # 1. Read current summary
    try:
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT chat_summary FROM customer_conversations WHERE conversation_id = %s",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            chat_summary = row[0] if row else ""
        updated_summary = move_active_to_resolved_in_summary(chat_summary)
    except Exception as e:
        logger.error("Failed to read chat_summary for resolve: %s", e)

    # 2. Persist resolved status to DB
    try:
        async with pool.connection() as conn:
            await conn.execute(
                """
                UPDATE customer_conversations
                SET status = 'active',
                    assigned_agent = NULL,
                    chat_summary = %s,
                    updated_at = NOW()
                WHERE conversation_id = %s
                """,
                (updated_summary, conversation_id),
            )
            await conn.commit()
    except Exception as e:
        logger.error("Failed to update conversation status in DB: %s", e)

    # 3. Reset LangGraph checkpoint — MUST happen so next customer message goes to AI
    config = {"configurable": {"thread_id": conversation_id}}
    try:
        await graph.aupdate_state(
            config,
            {
                "escalate_to_human": False,
                "chat_summary": updated_summary,
                "messages": [
                    SystemMessage(
                        content="SYSTEM: The human support agent has resolved this issue. Automated AI assistant is now re-enabled."
                    )
                ],
            },
        )
        logger.info(
            "✅ Escalation reset in LangGraph checkpoint for conversation %s",
            conversation_id,
        )
    except Exception as e:
        logger.error(
            "❌ CRITICAL: Failed to reset escalation state in graph for %s: %s — "
            "customer messages will still be routed to AI via DB status check.",
            conversation_id,
            e,
        )

    notification = json.dumps(
        {
            "event": "resolved",
            "content": "✅ *The support agent has resolved this issue. The AI assistant is ready to help you with other questions!*",
            "conversation_id": conversation_id,
        }
    )
    if redis_client:
        await redis_client.publish(f"chat:reply:{conversation_id}", notification)
        await redis_client.publish(
            "support:events",
            json.dumps({"event": "resolved", "conversation_id": conversation_id}),
        )
    logger.info("Conversation %s resolved via WS", conversation_id)
