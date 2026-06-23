"""
WebSocket chat handler route for streaming real-time responses.
"""

import asyncio
import json
import logging
import os
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler
import redis.asyncio as async_redis_raw

from app.middleware.rate_limit import check_rate_limit
from app.services.faq_response_cache import get_faq_response
from app.services.jwt_auth import verify_jwt
from app.services.support_handlers import (
    _handle_agent_join,
    _handle_agent_reply,
    _handle_agent_resolve,
    _handle_agent_typing,
)
from app.services.websocket_helpers import (
    _build_state_input,
    _check_escalation_status,
    _post_turn_escalation_check,
    _record_customer_bypass_message,
    _save_conversation_record,
    _stream_graph_events,
    _subscribe_pubsub,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/chat/ws")
async def chat_ws(
    websocket: WebSocket,
    token: str = Query(None),
    conversation_id: str = Query(None),
    role: str = Query("customer"),
    agent_name: str = Query(None),
):
    """
    WebSocket endpoint for real-time streaming of LangGraph agent responses.

    Maintains a persistent connection across multiple conversation turns.
    Each received JSON message triggers a full graph invocation and streams
    back token chunks and tool progress events.

    Message format (client → server):
        {"query": "...", "conversation_id": "...", "image_base64": "..."}

    Event format (server → client):
        {"type": "token",      "content": "..."}   — LLM token chunk
        {"type": "tool_start", "name": "...",  "inputs": "..."} — tool call
        {"type": "tool_end",   "name": "...",  "output": "..."} — tool result
        {"type": "end"}                             — turn complete
        {"type": "error",      "message": "..."}   — error
        {"type": "escalated_ack"}                   — message saved, AI bypassed
    """
    # ── 1. Authenticate ───────────────────────────────────────────────────────
    customer_id = None
    if role == "customer":
        if not token:
            await websocket.close(code=1008, reason="Missing token")
            return
        try:
            payload = verify_jwt(token)
            customer_id = payload["sub"]
        except Exception as e:
            logger.error("JWT verification failed: %s", e)
            await websocket.close(code=1008, reason="Invalid token")
            return
    else:
        if not conversation_id:
            await websocket.close(
                code=1008, reason="Missing conversation_id for support role"
            )
            return

    await websocket.accept()

    app_state = websocket.scope["app"].state

    # ── 2. Handle Support Role ────────────────────────────────────────────────
    if role == "support":
        logger.info(
            "Support WebSocket accepted for agent=%s, conversation_id=%s",
            agent_name,
            conversation_id,
        )
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        pubsub_redis = async_redis_raw.from_url(redis_url, decode_responses=True)
        pubsub, listener_task = await _subscribe_pubsub(
            pubsub_redis, conversation_id, websocket, app_state, role="support"
        )
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                except WebSocketDisconnect:
                    logger.info(
                        "Support WebSocket disconnected for conversation_id=%s",
                        conversation_id,
                    )
                    break
                except Exception as e:
                    logger.error("Error receiving support message: %s", e)
                    break

                event_type = data.get("type")
                if event_type == "agent_reply":
                    content = data.get("content")
                    if content:
                        await _handle_agent_reply(
                            conversation_id=conversation_id,
                            message=content,
                            agent_name=agent_name or "Support Agent",
                            app_state=app_state,
                        )
                elif event_type == "agent_typing":
                    await _handle_agent_typing(
                        conversation_id=conversation_id,
                        agent_name=agent_name or "Support Agent",
                        app_state=app_state,
                    )
                elif event_type == "join":
                    await _handle_agent_join(
                        conversation_id=conversation_id,
                        agent_name=agent_name or "Support Agent",
                        app_state=app_state,
                    )
                elif event_type == "resolve":
                    await _handle_agent_resolve(
                        conversation_id=conversation_id,
                        app_state=app_state,
                    )
                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Unsupported event type: {event_type}",
                        }
                    )
        finally:
            if listener_task:
                listener_task.cancel()
            if pubsub:
                await pubsub.unsubscribe()
                await pubsub.aclose()
            await pubsub_redis.aclose()
            logger.info(
                "Support WebSocket cleaned up for conversation_id=%s", conversation_id
            )
        return

    # ── 3. Customer Role ──────────────────────────────────────────────────────
    logger.info("WebSocket accepted for customer_id=%s", customer_id)

    # ── Subscribe to Redis Pub/Sub for human agent messages ───────────────────
    # Use a dedicated Redis connection — pub/sub must not share a connection
    # with command-based Redis operations.
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    pubsub_redis = async_redis_raw.from_url(redis_url, decode_responses=True)
    pubsub: async_redis_raw.client.PubSub | None = None
    listener_task: asyncio.Task | None = None

    # If conversation_id is known at connect time, subscribe immediately
    if conversation_id:
        pubsub, listener_task = await _subscribe_pubsub(
            pubsub_redis, conversation_id, websocket, app_state, role="customer"
        )

    try:
        # ── 3. Main message loop — one iteration per user turn ─────────────────
        while True:

            # ── 3a. Receive next message from the client ───────────────────────
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=600)
            except asyncio.TimeoutError:
                logger.info(
                    "WebSocket inactivity timeout for customer_id=%s", customer_id
                )
                try:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Connection closed due to 10 minutes of inactivity. Type a message to reconnect.",
                        }
                    )
                    await websocket.close(code=1000, reason="Inactivity timeout")
                except Exception:
                    pass
                break
            except WebSocketDisconnect:
                logger.info(
                    "WebSocket disconnected (receive) for customer_id=%s", customer_id
                )
                break
            except Exception as e:
                logger.error("Error receiving message: %s", e)
                break

            query: str = data.get("query", "")
            thread_id: str = data.get("conversation_id") or customer_id
            image_base64: str | None = data.get("image_base64")

            if not query:
                await websocket.send_json({"type": "error", "message": "Empty query."})
                continue

            # ── 3b. Rate limiting ──────────────────────────────────────────────
            redis_client = getattr(app_state, "redis", None)
            if await check_rate_limit(redis_client, customer_id):
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Too Many Requests. Please wait a minute before trying again.",
                    }
                )
                continue

            # ── 3c. Persist conversation record ───────────────────────────────
            try:
                await _save_conversation_record(
                    app_state.pool, thread_id, customer_id, query
                )
            except Exception as e:
                logger.error("DB error saving conversation: %s", e)
                await websocket.send_json(
                    {"type": "error", "message": "Failed to save conversation."}
                )
                continue

            # ── 3d. Lazy Pub/Sub subscription ─────────────────────────────────
            # If the thread_id wasn't known at connection time, subscribe now
            if pubsub is None:
                pubsub, listener_task = await _subscribe_pubsub(
                    pubsub_redis, thread_id, websocket, app_state
                )

            # ── 3e. Escalation bypass ─────────────────────────────────────────
            # If a human agent has taken over, save the message without invoking AI
            is_escalated = await _check_escalation_status(
                app_state.pool, thread_id, app_state.graph, redis_client
            )
            if is_escalated:
                try:
                    await _record_customer_bypass_message(
                        thread_id,
                        query,
                        image_base64,
                        customer_id,
                        app_state.graph,
                        redis_client,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to save human message in escalated state: %s", e
                    )
                logger.info(
                    "👤 Escalated message saved | customer=%s | query='%s'",
                    customer_id,
                    query,
                )
                # Remove the pending *Thinking...* UI indicator since AI is bypassed
                try:
                    await websocket.send_json({"type": "escalated_ack"})
                except (WebSocketDisconnect, Exception):
                    break
                continue

            # ── 3f. Application-level FAQ response cache ───────────────────────
            # If we have a cached response, bypass the graph entirely
            cached_response = (
                await get_faq_response(customer_id, query) if not image_base64 else None
            )
            if cached_response:
                logger.info(
                    "🟢 FAQ RESPONSE CACHE HIT (Bypass Graph) | customer=%s | query='%s'",
                    customer_id,
                    query,
                )
                try:
                    config = RunnableConfig(configurable={"thread_id": thread_id})
                    await app_state.graph.aupdate_state(
                        config,
                        {
                            "messages": [
                                HumanMessage(content=query),
                                AIMessage(content=cached_response, name="faq"),
                            ],
                            "customer_id": customer_id,
                            "query": query,
                        },
                    )
                except Exception as e:
                    logger.error("Failed to update state for cached FAQ: %s", e)

                await websocket.send_json({"type": "token", "content": cached_response})
                await websocket.send_json({"type": "end"})
                continue

            logger.info(
                "WS graph invoke | customer_id=%s | thread_id=%s",
                customer_id,
                thread_id,
            )

            # ── 3g. Build graph config with observability callbacks ────────────
            langfuse_handler = CallbackHandler()

            config = RunnableConfig(
                configurable={"thread_id": thread_id},
                callbacks=[langfuse_handler],
            )

            # ── 3h. Crash recovery / fresh state ──────────────────────────────
            state_input = await _build_state_input(
                query, customer_id, image_base64, app_state.graph, config
            )

            # ── 3i. Stream graph events to client ─────────────────────────────
            try:
                await _stream_graph_events(
                    app_state.graph, state_input, config, websocket
                )

                logger.info(
                    "Turn complete | customer_id=%s | thread_id=%s",
                    customer_id,
                    thread_id,
                )

                # ── 3j. Post-turn escalation check ────────────────────────────
                await _post_turn_escalation_check(
                    app_state.graph, config, app_state.pool, thread_id, redis_client
                )

            except WebSocketDisconnect:
                logger.info(
                    "Client disconnected mid-stream for customer_id=%s", customer_id
                )
                break
            except asyncio.TimeoutError:
                logger.error("Graph execution timed out.")
                try:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Request timed out. Please try again.",
                        }
                    )
                except Exception:
                    break
            except Exception as e:
                logger.error("Graph streaming error: %s", e, exc_info=True)
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except Exception:
                    break

    finally:
        # ── Cleanup on disconnect ──────────────────────────────────────────────
        # Publish offline event so the support dashboard updates presence status
        if conversation_id:
            redis_ref = getattr(app_state, "redis", None) or pubsub_redis
            try:
                await redis_ref.delete(f"presence:{conversation_id}")
                await redis_ref.publish(
                    "support:events",
                    json.dumps(
                        {
                            "event": "customer_offline",
                            "conversation_id": conversation_id,
                        }
                    ),
                )
                # Legacy event name for backward compatibility
                await redis_ref.publish(
                    "support:events",
                    json.dumps(
                        {
                            "event": "customer_disconnected",
                            "conversation_id": conversation_id,
                        }
                    ),
                )
            except Exception as e:
                logger.error("Failed to publish customer offline event: %s", e)

        # Cancel Pub/Sub listener and close the dedicated Redis connection
        if listener_task:
            listener_task.cancel()
        if pubsub:
            await pubsub.unsubscribe()
            await pubsub.aclose()
        await pubsub_redis.aclose()
