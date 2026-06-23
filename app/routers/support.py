"""
Support agent dashboard and interaction routes.
"""

import asyncio
import json
import logging
import os
import pathlib
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import redis.asyncio as async_redis

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/support/stream")
async def support_stream(request: Request):
    """
    SSE endpoint for the Support Dashboard to receive real-time event updates.

    Subscribes to the `support:events` Redis Pub/Sub channel and forwards events
    as Server-Sent Events.  Sends a `: keep-alive` comment every 30 seconds to
    prevent proxy timeouts.

    Events published to this stream include:
        new_escalation, customer_online, customer_offline, customer_message,
        agent_joined, agent_reply, agent_unassigned, resolved.
    """
    redis_client = getattr(request.app.state, "redis", None)
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not configured")

    shutdown_event: asyncio.Event = getattr(
        request.app.state, "shutdown_event", asyncio.Event()
    )

    # Use a dedicated Redis connection for Pub/Sub
    pubsub_redis = async_redis.Redis.from_url(os.getenv("REDIS_URL"))
    pubsub = pubsub_redis.pubsub()
    await pubsub.subscribe("support:events")

    # Feed messages into a queue so the generator remains cancellable
    queue: asyncio.Queue = asyncio.Queue()

    async def _reader():
        """Read messages from Pub/Sub and put them in the queue."""
        try:
            while not shutdown_event.is_set():
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=0.5
                )
                if msg:
                    await queue.put(msg["data"].decode("utf-8"))
                else:
                    await queue.put(None)  # heartbeat
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe()
            await pubsub.aclose()
            await pubsub_redis.aclose()

    reader_task = asyncio.create_task(_reader())

    async def event_generator():
        """Yield SSE events from the queue until shutdown or client disconnect."""
        try:
            while True:
                get_task = asyncio.ensure_future(queue.get())
                shutdown_task = asyncio.ensure_future(shutdown_event.wait())
                done, pending = await asyncio.wait(
                    [get_task, shutdown_task],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=30,  # Send keep-alive every 30s
                )
                for t in pending:
                    t.cancel()

                if shutdown_event.is_set():
                    break
                if await request.is_disconnected():
                    break

                if get_task in done:
                    data = get_task.result()
                    yield f"data: {data}\n\n" if data else ": keep-alive\n\n"
                else:
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            reader_task.cancel()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/support/conversations")
async def list_escalated_conversations(request: Request):
    """
    List all conversations with status='escalated' for the support dashboard.

    Also fetches each conversation's rolling chat summary from the LangGraph
    state so support agents have context before joining.

    Returns:
        A list of conversation dicts including summary and assigned_agent.
    """
    pool = request.app.state.pool
    graph = request.app.state.graph

    async with pool.connection() as conn:
        cursor = await conn.execute("""
            SELECT conversation_id, customer_id, title, updated_at, assigned_agent
            FROM customer_conversations
            WHERE status = 'escalated'
            ORDER BY updated_at DESC
            """)
        rows = await cursor.fetchall()

    conversations = []
    for row in rows:
        conv_id = row["conversation_id"]
        chat_summary = ""
        try:
            state = await graph.aget_state({"configurable": {"thread_id": conv_id}})
            chat_summary = state.values.get("chat_summary", "")
        except Exception:
            pass

        conversations.append(
            {
                "conversation_id": conv_id,
                "customer_id": row["customer_id"],
                "title": row["title"],
                "updated_at": (
                    row["updated_at"].isoformat() if row["updated_at"] else None
                ),
                "assigned_agent": row.get("assigned_agent"),
                "chat_summary": chat_summary,
            }
        )

    return conversations


@router.get("/support/conversations/{conversation_id}/history")
async def get_support_conversation_history(conversation_id: str, request: Request):
    """
    Fetch the full message history for a conversation for the support agent dashboard.

    Filters out tool messages and pure tool-call dispatcher messages so the
    agent sees a clean conversation view.  Also checks customer presence (online/offline).

    Returns:
        Dict with `messages` (list of {role, content, name}) and `is_online` (bool).
    """
    graph = request.app.state.graph

    # ── 1. Read graph state ───────────────────────────────────────────────────
    try:
        config = {"configurable": {"thread_id": conversation_id}}
        state = await graph.aget_state(config)
        messages = state.values.get("messages", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch state: {e}")

    # ── 2. Format messages (skip tool messages and empty AI dispatchers) ───────
    formatted = []
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        if msg_type == "tool":
            continue
        if msg_type == "ai" and getattr(msg, "tool_calls", None) and not msg.content:
            continue

        role = "user" if msg_type == "human" else "assistant"
        name = getattr(msg, "name", None)

        content_str = ""
        if msg.content:
            if isinstance(msg.content, list):
                text_parts = [
                    (
                        item.get("text", "")
                        if isinstance(item, dict) and item.get("type") == "text"
                        else str(item) if isinstance(item, str) else ""
                    )
                    for item in msg.content
                ]
                content_str = " ".join(text_parts)
            else:
                content_str = str(msg.content)

        formatted.append({"role": role, "content": content_str, "name": name})

    # ── 3. Check customer presence ────────────────────────────────────────────
    redis_client = getattr(request.app.state, "redis", None)
    is_online = False
    if redis_client:
        try:
            val = await redis_client.get(f"presence:{conversation_id}")
            is_online = val in (b"online", "online")
        except Exception:
            pass

    return {"messages": formatted, "is_online": is_online}


@router.get("/support/dashboard", response_class=HTMLResponse)
async def support_dashboard():
    """Serve the human support agent dashboard HTML page."""
    template_path = (
        pathlib.Path(__file__).parent.parent / "templates" / "support_dashboard.html"
    )
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard template not found")
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
