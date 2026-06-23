"""
WebSocket helper functions for handling client connections, Pub/Sub, and graph execution.
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from fastapi import WebSocket
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import observe
import redis.asyncio as async_redis_raw

from app.utils.summary import move_active_to_resolved_in_summary

logger = logging.getLogger(__name__)


async def _redis_pubsub_listener(
    pubsub: async_redis_raw.client.PubSub,
    websocket: WebSocket,
    role: str = "customer",
) -> None:
    """
    Background task that forwards messages from Redis to the client WebSocket.

    Listens indefinitely on the Redis Pub/Sub channel `chat:reply:<thread_id>` and
    passes event payloads directly to the WebSocket as JSON.

    Supported events: `agent_joined`, `agent_reply`, `resolved`, `agent_typing`,
    `customer_message`, `customer_typing`, `customer_online`, `customer_offline`, `customer_disconnected`.

    Args:
        pubsub: An active Redis Pub/Sub subscription object.
        websocket: The WebSocket connection.
        role: Either "customer" or "support" role.
    """
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                payload = json.loads(message["data"])
            except (json.JSONDecodeError, TypeError):
                continue

            event_type = payload.get("event")
            if role == "support":
                if event_type in (
                    "agent_joined",
                    "agent_reply",
                    "resolved",
                    "agent_typing",
                    "customer_message",
                    "customer_typing",
                    "customer_online",
                    "customer_offline",
                    "customer_disconnected",
                ):
                    await websocket.send_json(payload)
            else:
                if event_type in (
                    "agent_joined",
                    "agent_reply",
                    "resolved",
                    "agent_typing",
                ):
                    await websocket.send_json(payload)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.warning("Pub/Sub listener stopped for role=%s: %s", role, e)


@observe(name="customer_escalated_message", as_type="generation")
async def _record_customer_bypass_message(
    thread_id: str,
    query: str,
    image_base64: str | None,
    customer_id: str,
    graph,
    redis_client,
) -> None:
    """
    Append a user message to the graph state without invoking the AI pipeline.

    Called when a conversation is in 'escalated' status so the customer's
    messages are stored in history and visible to the human agent, but the
    AI is not triggered.  Also publishes a `customer_message` event to the
    support dashboard via Redis.

    Args:
        thread_id: The conversation thread ID.
        query: The user's text message.
        image_base64: Optional base64 image string.
        customer_id: The authenticated customer ID.
        graph: The compiled LangGraph application.
        redis_client: Async Redis client for publishing events.
    """
    config = RunnableConfig(configurable={"thread_id": thread_id})

    # Save only text to the checkpoint.
    # Do NOT embed the raw base64 image in the graph state — large payloads
    # (1–5 MB) passed through a remote PostgreSQL checkpointer connection
    # cause [Errno 32] Broken pipe.  The image is forwarded to the support
    # dashboard via a separate storage path (URL reference), not raw bytes.
    text_content = query
    if image_base64:
        text_content = f"{query}\n[Image attached]" if query else "[Image attached]"

    await graph.aupdate_state(
        config,
        {
            "messages": [HumanMessage(content=text_content)],
            # Do NOT reset escalate_to_human — the conversation is still escalated.
            # It will be reset by /resolve when the support agent resolves the issue.
            "customer_id": customer_id,
            "query": query,
        },
    )

    if redis_client:
        payload: dict = {
            "event": "customer_message",
            "conversation_id": thread_id,
            "content": query,
        }
        if image_base64:
            # Flag that an image was included — do NOT embed raw base64 bytes.
            # Redis pub/sub is not designed for large binary payloads and will
            # drop or truncate messages exceeding the server's client-output-buffer
            # limit, again causing Broken Pipe errors.
            payload["has_image"] = True

        # Publish to the specific conversation channel
        await redis_client.publish(
            f"chat:reply:{thread_id}",
            json.dumps(payload),
        )
        # Publish to the global support channel for dashboard updates
        await redis_client.publish(
            "support:events",
            json.dumps({"event": "customer_message", "conversation_id": thread_id}),
        )


async def _subscribe_pubsub(
    pubsub_redis: async_redis_raw.Redis,
    thread_id: str,
    websocket: WebSocket,
    app_state,
    role: str = "customer",
) -> tuple:
    """
    Subscribe to the Redis Pub/Sub channel for a conversation thread and start
    the listener background task.  Also publishes a `customer_online` presence event.

    Args:
        pubsub_redis: A dedicated Redis connection for Pub/Sub (not shared with the pool).
        thread_id: The conversation thread ID.
        websocket: The customer's WebSocket connection.
        app_state: The FastAPI application state.
        role: The role connecting to the websocket.

    Returns:
        A tuple of (pubsub, listener_task).
    """
    pubsub = pubsub_redis.pubsub()
    await pubsub.subscribe(f"chat:reply:{thread_id}")
    listener_task = asyncio.create_task(_redis_pubsub_listener(pubsub, websocket, role))

    # Publish presence event so the support dashboard shows the customer as online
    redis_client = getattr(app_state, "redis", None)
    if redis_client:
        try:
            await redis_client.set(f"presence:{thread_id}", "online")
            await redis_client.publish(
                "support:events",
                json.dumps({"event": "customer_online", "conversation_id": thread_id}),
            )
        except Exception:
            pass

    return pubsub, listener_task


async def _save_conversation_record(
    pool, thread_id: str, customer_id: str, query: str
) -> None:
    """
    Upsert a conversation record in the database for the current turn.

    Uses an INSERT ... ON CONFLICT DO UPDATE so the first message creates the
    record and all subsequent messages update the `updated_at` timestamp.

    Args:
        pool: Async Postgres connection pool.
        thread_id: The conversation thread ID.
        customer_id: The authenticated customer ID.
        query: The user's message (truncated to 30 chars for the title).
    """
    title = (query[:30] + "...") if len(query) > 30 else query
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO customer_conversations (conversation_id, customer_id, title)
            VALUES (%s, %s, %s)
            ON CONFLICT (conversation_id) DO UPDATE SET updated_at = NOW()
            """,
            (thread_id, customer_id, title),
        )


async def _check_escalation_status(
    pool, thread_id: str, graph=None, redis_client=None
) -> bool:
    """
    Check whether a conversation is currently in 'escalated' status,
    automatically resolving it if it has timed out due to 10 minutes of inactivity.

    Args:
        pool: Async Postgres connection pool.
        thread_id: The conversation thread ID.
        graph: The compiled LangGraph application (optional).
        redis_client: Async Redis client (optional).

    Returns:
        True if the conversation is active and escalated, False otherwise.
    """
    try:
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT status, updated_at, chat_summary FROM customer_conversations WHERE conversation_id = %s",
                (thread_id,),
            )
            row = await cursor.fetchone()
            if not row:
                return False

            # Support row indexing both as dict and tuple
            status = row.get("status") if hasattr(row, "get") else row[0]
            updated_at = row.get("updated_at") if hasattr(row, "get") else row[1]
            chat_summary = (
                row.get("chat_summary") if hasattr(row, "get") else row[2]
            ) or ""

            if status != "escalated":
                return False

            # Check if 10 minutes of inactivity has passed (only if graph is provided to resolve)
            if graph is not None:
                now = datetime.now(timezone.utc)
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)

                elapsed_seconds = (now - updated_at).total_seconds()
                if elapsed_seconds > 600:
                    logger.info(
                        "⌛ Escalated conversation %s timed out after %s seconds of inactivity. Re-enabling AI.",
                        thread_id,
                        elapsed_seconds,
                    )

                    # 1. Update status to 'active' and assigned_agent to NULL in DB
                    updated_summary = move_active_to_resolved_in_summary(chat_summary)
                    await conn.execute(
                        """
                        UPDATE customer_conversations
                        SET status = 'active',
                            assigned_agent = NULL,
                            chat_summary = %s,
                            updated_at = NOW()
                        WHERE conversation_id = %s
                        """,
                        (updated_summary, thread_id),
                    )
                    await conn.commit()

                    # 2. Reset LangGraph state escalation flag and append system message
                    config = {"configurable": {"thread_id": thread_id}}
                    try:
                        await graph.aupdate_state(
                            config,
                            {
                                "escalate_to_human": False,
                                "chat_summary": updated_summary,
                                "messages": [
                                    SystemMessage(
                                        content="SYSTEM: The conversation was automatically resolved due to 10 minutes of customer/agent inactivity. Automated AI assistant is now re-enabled."
                                    )
                                ],
                            },
                        )
                    except Exception as e:
                        logger.error(
                            "Failed to reset escalation state on timeout: %s", e
                        )

                    # 3. Publish to Redis Pub/Sub
                    if redis_client:
                        notification = json.dumps(
                            {
                                "event": "resolved",
                                "content": "✅ *The conversation was automatically returned to the AI assistant due to inactivity.*",
                                "conversation_id": thread_id,
                            }
                        )
                        await redis_client.publish(
                            f"chat:reply:{thread_id}", notification
                        )
                        await redis_client.publish(
                            "support:events",
                            json.dumps(
                                {"event": "resolved", "conversation_id": thread_id}
                            ),
                        )

                    return False

            return True
    except Exception as e:
        logger.warning("Failed to check escalation status or resolve timeout: %s", e)
        return False


async def _build_state_input(
    query: str,
    customer_id: str,
    image_base64: str | None,
    graph,
    config: RunnableConfig,
) -> dict | None:
    """
    Build the initial state input dict for the graph, handling crash recovery.

    If the graph has pending nodes from a previous crashed turn, we resume from
    the existing checkpoint rather than injecting a duplicate HumanMessage.

    Args:
        query: The user's message text.
        customer_id: The authenticated customer ID.
        image_base64: Optional base64 image string.
        graph: The compiled LangGraph application.
        config: The current turn's RunnableConfig.

    Returns:
        A state dict if starting fresh, or None if resuming a crashed turn.
    """
    state_input = {
        "query": query,
        "customer_id": customer_id,
        "messages": [HumanMessage(content=query)],
        "escalate_to_human": False,
        "image_base64": image_base64 or None,
        # Reset ephemeral image safety state so a previous block doesn't leak
        "image_safety_warning": None,
        "image_is_safe": None,
    }

    try:
        current_state = await graph.aget_state(config)
        if current_state.next:
            logger.info(
                "Crash recovery: pending nodes=%s. Injecting fresh query and resuming.",
                current_state.next,
            )
            await graph.aupdate_state(
                config,
                {
                    "customer_id": customer_id,
                    "query": query,
                    # Always clear escalation on crash-resume so supervisor doesn't
                    # short-circuit back into human-handoff mode
                    "escalate_to_human": False,
                },
            )
            # Return None → LangGraph resumes from checkpoint, no duplicate HumanMessage
            return None
    except Exception as e:
        logger.warning("Could not check/resume graph state: %s. Starting fresh.", e)

    return state_input


async def _stream_graph_events(
    graph, state_input: dict | None, config: RunnableConfig, websocket: WebSocket
) -> None:
    """
    Stream LangGraph events to the WebSocket and emit `end` when complete.

    Handles two event types:
    - `on_chat_model_stream`: streams token chunks from FAQ, Order, and Synthesizer nodes.
    - `on_tool_start` / `on_tool_end`: sends tool progress indicators to the client.

    If multiple agents are pending, only streams the Synthesizer's tokens (the
    individual sub-agent streams would be confusing and redundant).

    If no tokens were streamed (e.g. pure cache hit or direct supervisor reply),
    reads the last AI message from the final graph state and sends it as a token.

    Args:
        graph: The compiled LangGraph application.
        state_input: The initial state dict, or None to resume from checkpoint.
        config: The current turn's RunnableConfig.
        websocket: The customer's WebSocket connection.
    """
    tokens_streamed = False
    pending_agents_count = -1  # Fetched lazily on first streaming event

    async for event in graph.astream_events(state_input, config=config, version="v2"):
        event_type = event["event"]

        # ── Token streaming ───────────────────────────────────────────────────
        if event_type == "on_chat_model_stream":
            metadata = event.get("metadata", {})
            node = metadata.get("langgraph_node")

            # Only stream from these nodes; skip supervisor, summarizer, etc.
            if node not in ["faq", "order", "synthesizer"]:
                continue

            # Lazily determine how many agents are running this turn
            if pending_agents_count == -1:
                current_state = await graph.aget_state(config)
                pending_agents_count = len(
                    current_state.values.get("pending_agents", [])
                )

            # When multiple agents run, only stream the synthesizer to avoid confusion
            if pending_agents_count > 1 and node != "synthesizer":
                continue

            chunk = event["data"]["chunk"]
            content = chunk.content

            # Normalise Anthropic multimodal content format (list of blocks → string)
            if isinstance(content, list):
                content = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )

            if content:
                tokens_streamed = True
                await websocket.send_json({"type": "token", "content": content})

        # ── Tool start/end indicators ─────────────────────────────────────────
        elif event_type == "on_tool_start":
            await websocket.send_json(
                {
                    "type": "tool_start",
                    "name": event["name"],
                    "inputs": str(event["data"].get("input", {})),
                }
            )
        elif event_type == "on_tool_end":
            await websocket.send_json(
                {
                    "type": "tool_end",
                    "name": event["name"],
                    "output": str(event["data"].get("output", "")),
                }
            )

    # ── Fallback for cache hits and direct supervisor responses ───────────────
    # When no tokens were streamed, read the last AI message from the state.
    if not tokens_streamed:
        current_state = await graph.aget_state(config)
        messages = current_state.values.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if getattr(last_msg, "type", "") == "ai" and last_msg.content:
                await websocket.send_json(
                    {"type": "token", "content": last_msg.content}
                )

    # Signal the client that the turn is complete
    await websocket.send_json({"type": "end"})


async def _post_turn_escalation_check(
    graph, config: RunnableConfig, pool, thread_id: str, redis_client
) -> None:
    """
    After the graph finishes, check if it set `escalate_to_human=True`.

    If so, update the conversation status to 'escalated' in the database and
    publish a `new_escalation` event to the support dashboard.

    Args:
        graph: The compiled LangGraph application.
        config: The current turn's RunnableConfig.
        pool: Async Postgres connection pool.
        thread_id: The conversation thread ID.
        redis_client: Async Redis client for publishing events.
    """
    try:
        final_state = await graph.aget_state(config)
        if final_state.values.get("escalate_to_human"):
            async with pool.connection() as conn:
                await conn.execute(
                    "UPDATE customer_conversations SET status = 'escalated' WHERE conversation_id = %s",
                    (thread_id,),
                )
            if redis_client:
                await redis_client.publish(
                    "support:events",
                    json.dumps(
                        {"event": "new_escalation", "conversation_id": thread_id}
                    ),
                )
            logger.info("🚨 Conversation escalated to human | thread_id=%s", thread_id)
    except Exception as e:
        logger.warning("Could not check/update escalation status: %s", e)
