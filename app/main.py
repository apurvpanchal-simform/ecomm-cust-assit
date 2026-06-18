"""
Main FastAPI application entry point for the E-Commerce Customer Assistant.

Sets up the LangGraph pipeline, authentication, and chat endpoints.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, List

import redis
import redis.asyncio as async_redis
import redis.asyncio as async_redis_raw
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from langchain_community.cache import RedisCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langfuse import observe
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.db.customers import (
    CustomerService,
)
from app.db.supabase import get_supabase_client
from app.graph.builder import compile_graph
from app.graph.checkpointer import AsyncDualCheckpointer
from app.middleware.rate_limit import check_rate_limit, rate_limit_customer, RateLimitCallbackHandler
from app.middleware.distributed_lock import RedisTokenBucket
from app.schemas.api import (
    ConversationItem,
    LoginRequest,
    LoginResponse,
)
from app.services.faq_response_cache import get_faq_response
from app.services.jwt_auth import (
    generate_jwt,
    verify_jwt,
)

load_dotenv(override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for initializing global infrastructure.

    Sets up the AsyncPostgresSaver and AsyncRedisSaver for LangGraph checkpointing,
    ensures the required tables exist, and compiles the agentic graph.

    Args:
        app: The FastAPI application instance.
    """
    redis_url = os.getenv("REDIS_URL")
    supabase_db_url = os.getenv("SUPABASE_DB_URL")

    if not redis_url:
        raise RuntimeError("REDIS_URL environment variable is missing")

    if not supabase_db_url:
        raise RuntimeError("SUPABASE_DB_URL environment variable is missing")

    try:
        async with (
            AsyncConnectionPool(
                supabase_db_url,
                min_size=1,
                max_size=4,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                },
            ) as pool,
            async_redis.Redis.from_url(redis_url) as redis_client,
        ):
            logger.info("Initializing AsyncPostgresSaver...")
            postgres_saver = AsyncPostgresSaver(pool)
            await postgres_saver.setup()

            # Ensure custom checkpoint_state_logs table exists
            logger.info("Ensuring checkpoint_state_logs table exists...")
            async with pool.connection() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoint_state_logs (
                        id                   BIGSERIAL PRIMARY KEY,
                        conversation_id      TEXT NOT NULL,
                        checkpoint_id        TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        step_node            TEXT,
                        state_values         JSONB NOT NULL,
                        metadata             JSONB,
                        created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_checkpoint_state_logs_conversation 
                    ON checkpoint_state_logs(conversation_id);
                """)

            # The customer_conversations table is managed in app/db/schema.sql
            # Ensure escalation columns exist
            async with pool.connection() as conn:
                await conn.execute("""
                    ALTER TABLE customer_conversations
                    ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active';
                """)
                await conn.execute("""
                    ALTER TABLE customer_conversations
                    ADD COLUMN IF NOT EXISTS assigned_agent TEXT;
                """)
            logger.info("Ensured escalation columns (status, assigned_agent) exist.")

            app.state.pool = pool
            app.state.redis = redis_client
            app.state.shutdown_event = asyncio.Event()

            async def agent_inactivity_sweep():
                """Background task to periodically check for inactive human agents and unassign them."""
                while not app.state.shutdown_event.is_set():
                    try:
                        async with app.state.pool.connection() as conn:
                            cursor = await conn.execute("""
                                SELECT conversation_id, assigned_agent 
                                FROM customer_conversations 
                                WHERE status = 'escalated' 
                                  AND assigned_agent IS NOT NULL 
                                  AND updated_at < NOW() - INTERVAL '10 minutes'
                                """)
                            inactive_convs = await cursor.fetchall()
                            for conv in inactive_convs:
                                conv_id = conv["conversation_id"]
                                agent_name = conv["assigned_agent"]

                                await conn.execute(
                                    """
                                    UPDATE customer_conversations 
                                    SET assigned_agent = NULL, updated_at = NOW() 
                                    WHERE conversation_id = %s
                                    """,
                                    (conv_id,),
                                )

                                if hasattr(app.state, "redis") and app.state.redis:
                                    notification = json.dumps(
                                        {
                                            "event": "agent_reply",
                                            "agent_name": "System",
                                            "content": "⚠️ *We apologize for the delay. An agent will be with you shortly.*",
                                        }
                                    )
                                    await app.state.redis.publish(
                                        f"chat:reply:{conv_id}", notification
                                    )
                                    await app.state.redis.publish(
                                        "support:events",
                                        json.dumps(
                                            {
                                                "event": "agent_unassigned",
                                                "conversation_id": conv_id,
                                            }
                                        ),
                                    )

                                logger.info(
                                    logger.info(
                                        "Agent %s unassigned from %s due to inactivity.",
                                        agent_name,
                                        conv_id,
                                    )
                                )
                    except Exception as e:
                        logger.error("Error in agent_inactivity_sweep: %s", e)

                    try:
                        await asyncio.wait_for(
                            app.state.shutdown_event.wait(), timeout=60
                        )
                    except asyncio.TimeoutError:
                        pass

            sweep_task = asyncio.create_task(agent_inactivity_sweep())

            logger.info("Initializing AsyncRedisSaver...")
            redis_saver = AsyncRedisSaver(redis_client=redis_client)
            await redis_saver.asetup()

            class LoggingRedisCache(RedisCache):
                def _is_image_query(self, prompt: str) -> bool:
                    # If the prompt contains any of these indicators, we completely bypass the global cache.
                    indicators = [
                        "The user uploaded an image",
                        "image_safety_warning",
                        "The user's uploaded image was blocked",
                    ]
                    return any(ind in prompt for ind in indicators)

                def lookup(self, prompt: str, llm_string: str):
                    if self._is_image_query(prompt):
                        return None
                    hit = super().lookup(prompt, llm_string)
                    if hit:
                        logger.info("🟢 CACHE HIT! Returning cached response.")
                    else:
                        logger.info("🔴 CACHE MISS! Generating new response...")
                    return hit

                async def alookup(self, prompt: str, llm_string: str):
                    if self._is_image_query(prompt):
                        return None
                    hit = await super().alookup(prompt, llm_string)
                    if hit:
                        logger.info("🟢 CACHE HIT! Returning cached response.")
                    else:
                        logger.info("🔴 CACHE MISS! Generating new response...")
                    return hit

                def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
                    if self._is_image_query(prompt):
                        return
                    super().update(prompt, llm_string, return_val)

                async def aupdate(
                    self, prompt: str, llm_string: str, return_val: Any
                ) -> None:
                    if self._is_image_query(prompt):
                        return
                    await super().aupdate(prompt, llm_string, return_val)

            logger.info(
                "Enabling global LangChain LLM Redis EXACT MATCH cache with 1-hour TTL..."
            )
            sync_redis_client = redis.Redis.from_url(redis_url)
            set_llm_cache(LoggingRedisCache(redis_=sync_redis_client, ttl=3600))

            dual_checkpointer = AsyncDualCheckpointer(
                redis_saver=redis_saver,
                postgres_saver=postgres_saver,
                pool=pool,
            )

            app.state.graph = compile_graph(checkpointer=dual_checkpointer)

            logger.info(
                "Graph compiled successfully with Redis + Postgres async checkpointing."
            )

            yield
            # Signal all long-lived SSE streams to exit immediately
            # (must be set right here, before Uvicorn waits for connections to close)
            app.state.shutdown_event.set()
            app.state.is_shutting_down = True
            sweep_task.cancel()

    except Exception:
        logger.exception("Failed to initialize graph infrastructure.")
        raise


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.

    Returns:
        A dictionary with a simple 'ok' status.
    """
    return {"status": "ok"}


@app.post(
    "/auth/login",
    response_model=LoginResponse,
)
async def login(
    request: LoginRequest,
):
    """
    Authenticate a customer by email.

    Args:
        request: The login request containing the customer's email.

    Returns:
        A JWT access token upon successful authentication.

    Raises:
        HTTPException: If the customer is not found.
    """

    customer_service = CustomerService()

    customer = customer_service.get_customer_by_email(request.email)

    if not customer:
        raise HTTPException(
            status_code=401,
            detail="Customer not found",
        )

    token = generate_jwt(
        customer_id=customer["customer_id"],
        email=customer["email"],
    )

    return LoginResponse(access_token=token)


@app.get("/chat/conversations", response_model=List[ConversationItem])
async def list_conversations(
    request: Request,
    customer_id: str = Depends(rate_limit_customer),
):
    """
    Retrieve a list of past conversations for the authenticated customer.

    Args:
        request: The FastAPI request object.
        customer_id: The authenticated customer ID.

    Returns:
        A list of ConversationItem schemas containing metadata about past chats.
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


@app.get("/chat/history/{conversation_id}")
async def get_chat_history(
    conversation_id: str,
    request: Request,
    customer_id: str = Depends(rate_limit_customer),
):
    """
    Retrieve the full message history for a specific conversation thread.

    Validates ownership of the conversation and formats the stored LangGraph
    state messages into a clean user/assistant format.

    Args:
        conversation_id: The ID of the conversation to fetch.
        request: The FastAPI request object.
        customer_id: The authenticated customer ID.

    Returns:
        A dictionary containing the formatted message history.

    Raises:
        HTTPException: If the conversation is not found or not owned by the user.
    """
    async with request.app.state.pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT status FROM customer_conversations WHERE conversation_id = %s AND customer_id = %s",
            (conversation_id, customer_id),
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=403, detail="Conversation not found")

    config = {"configurable": {"thread_id": conversation_id}}
    state = await request.app.state.graph.aget_state(config)

    messages = state.values.get("messages", [])

    formatted_messages = []
    current_turn_ai_messages = []

    def flush_ai_messages():
        if current_turn_ai_messages:
            synth_msg = next(
                (
                    m
                    for m in current_turn_ai_messages
                    if getattr(m, "name", "") == "synthesizer"
                ),
                None,
            )
            final_msg = synth_msg if synth_msg else current_turn_ai_messages[-1]
            formatted_messages.append(
                {
                    "role": "assistant",
                    "content": str(final_msg.content),
                }
            )
            current_turn_ai_messages.clear()

    for msg in messages:
        if msg.type == "human":
            flush_ai_messages()
            if msg.content:
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
                    {
                        "role": "user",
                        "content": text_content,
                        "images": images,
                    }
                )
        elif msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
            current_turn_ai_messages.append(msg)

    flush_ai_messages()

    return {"status": row["status"], "messages": formatted_messages}


@app.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    request: Request,
    customer_id: str = Depends(rate_limit_customer),
):
    """Delete a specific conversation from history."""
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


@app.get("/orders")
async def get_orders(
    customer_id: str = Depends(rate_limit_customer),
):
    """Fetch raw user orders from Supabase."""
    supabase = await get_supabase_client()
    result = (
        await supabase.table("orders")
        .select("*")
        .eq("customer_id", customer_id)
        .order("ordered_at", desc=True)
        .execute()
    )
    return result.data


@app.get("/products")
async def get_products(
    customer_id: str = Depends(rate_limit_customer),
):
    """Fetch raw products from Supabase."""
    supabase = await get_supabase_client()
    result = await supabase.table("products").select("*").execute()
    return result.data


# ── Redis Pub/Sub Listener for Human Agent Messages ──────────────────────────


async def _redis_pubsub_listener(
    pubsub: async_redis_raw.client.PubSub,
    websocket: WebSocket,
) -> None:
    """Background task that forwards human agent messages from Redis to the user's WebSocket."""
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                payload = json.loads(message["data"])
            except (json.JSONDecodeError, TypeError):
                continue

            event_type = payload.get("event")
            _content = payload.get("content", "")

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
        logger.warning("Pub/Sub listener stopped: %s", e)


@observe(name="customer_escalated_message", as_type="generation")
async def _record_customer_bypass_message(
    thread_id: str,
    query: str,
    image_base64: str | None,
    customer_id: str,
    graph,
    redis_client,
):
    config = RunnableConfig(configurable={"thread_id": thread_id})
    msg_content = [{"type": "text", "text": query}]
    if image_base64:
        msg_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            }
        )
    await graph.aupdate_state(
        config,
        {
            "messages": [HumanMessage(content=msg_content)],
            # Do NOT reset escalate_to_human here — the conversation is still escalated.
            # It will be reset by the /resolve endpoint when the agent resolves.
            "customer_id": customer_id,
            "query": query,
        },
    )
    if redis_client:
        await redis_client.publish(
            "support:events",
            json.dumps({"event": "customer_message", "conversation_id": thread_id}),
        )


@app.websocket("/chat/ws")
async def chat_ws(
    websocket: WebSocket, token: str = Query(...), conversation_id: str = Query(None)
):
    """
    WebSocket endpoint for real-time streaming of LangGraph agent events.
    Maintains a persistent connection across multiple conversation turns.
    """
    # ── 1. Authenticate ──────────────────────────────────────────────────
    try:
        payload = verify_jwt(token)
        customer_id = payload["sub"]
    except Exception as e:
        logger.error("JWT verification failed: %s", e)
        await websocket.close(code=1008, reason="Invalid token")
        return

    await websocket.accept()
    logger.info("WebSocket accepted for customer_id=%s", customer_id)

    app_state = websocket.scope["app"].state

    # ── 1b. Subscribe to Redis Pub/Sub for human agent messages ───────────
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    pubsub_redis = async_redis_raw.from_url(redis_url, decode_responses=True)
    pubsub: async_redis_raw.client.PubSub | None = None
    listener_task: asyncio.Task | None = None

    if conversation_id:
        pubsub = pubsub_redis.pubsub()
        await pubsub.subscribe(f"chat:reply:{conversation_id}")
        listener_task = asyncio.create_task(_redis_pubsub_listener(pubsub, websocket))
        # Notify support dashboard that customer is now online
        redis_client_ref = getattr(app_state, "redis", None)
        if redis_client_ref:
            try:
                await redis_client_ref.set(f"presence:{conversation_id}", "online")
                await redis_client_ref.publish(
                    "support:events",
                    json.dumps(
                        {"event": "customer_online", "conversation_id": conversation_id}
                    ),
                )
            except Exception:
                pass

    # ── 2. Message loop — one iteration per user turn ────────────────────
    try:
        while True:
            # ── 2a. Receive ──────────────────────────────────────────────────
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=600)
            except asyncio.TimeoutError:
                logger.info("WebSocket closed due to inactivity for customer_id=%s", customer_id)
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
                logger.info("WebSocket disconnected (receive) for customer_id=%s", customer_id)
                break
            except Exception as e:
                logger.error("Error receiving message: %s", e)
                break  # unrecoverable — socket is broken

            query: str = data.get("query", "")
            thread_id: str = data.get("conversation_id") or customer_id
            image_base64: str | None = data.get("image_base64")

            if not query:
                await websocket.send_json({"type": "error", "message": "Empty query."})
                continue  # stay connected, wait for next message

            # ── 2b. Rate Limiting ───────────────────────────────────────────
            redis_client = getattr(app_state, "redis", None)
            if await check_rate_limit(redis_client, customer_id):
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Too Many Requests. Please wait a minute before trying again.",
                    }
                )
                continue

            # ── 2b. Persist conversation record ─────────────────────────────
            try:
                async with app_state.pool.connection() as conn:
                    await conn.execute(
                        """
                        INSERT INTO customer_conversations (conversation_id, customer_id, title)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (conversation_id) DO UPDATE SET updated_at = NOW()
                        """,
                        (
                            thread_id,
                            customer_id,
                            (query[:30] + "...") if len(query) > 30 else query,
                        ),
                    )
            except Exception as e:
                logger.error("DB error saving conversation: %s", e)
                await websocket.send_json(
                    {"type": "error", "message": "Failed to save conversation."}
                )
                continue  # not fatal — still try to run the graph

            # ── 2c. Subscribe to Pub/Sub channel for this thread (lazy) ──────
            if pubsub is None:
                pubsub = pubsub_redis.pubsub()
                await pubsub.subscribe(f"chat:reply:{thread_id}")
                listener_task = asyncio.create_task(
                    _redis_pubsub_listener(pubsub, websocket)
                )
                # Also mark customer as online now that we know the thread_id
                redis_client_lazy = getattr(app_state, "redis", None)
                if redis_client_lazy:
                    try:
                        await redis_client_lazy.set(f"presence:{thread_id}", "online")
                        await redis_client_lazy.publish(
                            "support:events",
                            json.dumps(
                                {"event": "customer_online", "conversation_id": thread_id}
                            ),
                        )
                    except Exception:
                        pass

            # ── 2d. Escalation bypass — route to human, skip AI ──────────────
            try:
                async with app_state.pool.connection() as conn:
                    cursor = await conn.execute(
                        "SELECT status FROM customer_conversations WHERE conversation_id = %s",
                        (thread_id,),
                    )
                    row = await cursor.fetchone()
                    is_escalated = row and row.get("status") == "escalated"
            except Exception:
                is_escalated = False

            if is_escalated:
                # Write user message to history without invoking the graph
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
                    logger.error("Failed to save human message in escalated state: %s", e)
                logger.info("👤 Escalated message saved | customer=%s | query='%s'", customer_id, query)
                # Remove the pending '*Thinking...*' message in the UI since AI is bypassed
                await websocket.send_json({"type": "escalated_ack"})
                continue

            # --- Application-level Response Cache Hit Check (Bypass Graph) ---
            cached_response = (
                await get_faq_response(customer_id, query) if not image_base64 else None
            )
            if cached_response:
                logger.info(
                    logger.info(
                        "🟢 FAQ RESPONSE CACHE HIT (Bypass Graph) | customer=%s | query='%s'",
                        customer_id,
                        query,
                    )
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
                    logger.error(
                        logger.info(
                            "Failed to update graph state for cached FAQ response: %s",
                            e,
                        )
                    )

                await websocket.send_json({"type": "token", "content": cached_response})
                await websocket.send_json({"type": "end"})
                continue

            logger.info(
                logger.info(
                    "WS graph invoke | customer_id=%s | thread_id=%s",
                    customer_id,
                    thread_id,
                )
            )

            # ── 2c. Build graph config ───────────────────────────────────────
            langfuse_handler = CallbackHandler()
            
            redis_client = getattr(app_state, "redis", None)
            # 25 RPM for safety (25 tokens max burst, 25 tokens / 60 seconds fill rate)
            token_bucket = RedisTokenBucket(redis_client, "llm_rate_limit", capacity=25, fill_rate=25/60) if redis_client else None
            rate_limiter_callback = RateLimitCallbackHandler(token_bucket)

            config = RunnableConfig(
                configurable={"thread_id": thread_id},
                callbacks=[langfuse_handler, rate_limiter_callback],
            )

            # ── 2d. Crash recovery ───────────────────────────────────────────
            # Build state_input first; may be set to None if we are resuming a crash.
            state_input = {
                "query": query,
                "customer_id": customer_id,
                "messages": [HumanMessage(content=query)],
                "escalate_to_human": False,
                "image_base64": image_base64 or None,
                # Reset ephemeral image state so previous-turn safety blocks
                # don't leak into the current turn.
                "image_safety_warning": None,
                "image_is_safe": None,
            }

            try:
                current_state = await app_state.graph.aget_state(config)
                if current_state.next:
                    logger.info(
                        "Crash recovery: pending nodes=%s. Injecting fresh query and resuming.",
                        current_state.next,
                    )
                    await app_state.graph.aupdate_state(
                        config,
                        {"customer_id": customer_id, "query": query},
                    )
                    # Pass None so LangGraph resumes from the pending checkpoint
                    # rather than injecting a duplicate HumanMessage.
                    state_input = None
            except Exception as e:
                # Non-fatal: log and proceed with state_input as-is
                logger.warning(
                    logger.info(
                        "Could not check/resume graph state: %s. Starting fresh.", e
                    )
                )

            # ── 2e. Stream graph events ──────────────────────────────────────
            # This is isolated in its own try/except so that ANY error during
            # streaming sends {"type":"error"} and then CONTINUES the while loop
            # instead of propagating up to the outer scope and closing the socket.
            try:
                tokens_streamed = False
                pending_agents_count = -1

                # ── Stream Events ─────────────────
                if True:
                    async for event in app_state.graph.astream_events(
                        state_input,
                        config=config,
                        version="v2",
                    ):
                        event_type = event["event"]

                        if event_type == "on_custom_event":
                            if event.get("name") in ["queue_wait_start", "queue_wait_end"]:
                                # We emit these as tool_start and tool_end to render smoothly in chainlit
                                await websocket.send_json({
                                    "type": "tool_start" if event["name"] == "queue_wait_start" else "tool_end",
                                    "name": "Queue Wait",
                                    "inputs" if event["name"] == "queue_wait_start" else "output": event["data"].get("message", "...")
                                })
                                continue
    
                        if event_type == "on_chat_model_stream":
                            metadata = event.get("metadata", {})
                            node = metadata.get("langgraph_node")
                            if node not in ["faq", "order", "synthesizer"]:
                                continue
    
                            # Fetch the state once to determine how many agents are running
                            if pending_agents_count == -1:
                                current_state = await app_state.graph.aget_state(config)
                                pending_agents_count = len(
                                    current_state.values.get("pending_agents", [])
                                )
    
                            # If multiple agents are running, ONLY stream the final synthesizer
                            if pending_agents_count > 1 and node != "synthesizer":
                                continue
    
                            chunk = event["data"]["chunk"]
                            # chunk.content can be a string or a list of content blocks
                            content = chunk.content
                            if isinstance(content, list):
                                # Extract text from content blocks (e.g. Anthropic format)
                                content = "".join(
                                    (
                                        block.get("text", "")
                                        if isinstance(block, dict)
                                        else str(block)
                                    )
                                    for block in content
                                )
                            if content:
                                tokens_streamed = True
                                await websocket.send_json(
                                    {"type": "token", "content": content}
                                )
    
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

                # If no tokens were streamed (e.g. cache hit or direct supervisor response),
                # send the last AI message from the final graph state.
                if not tokens_streamed:
                    current_state = await app_state.graph.aget_state(config)
                    messages = current_state.values.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if getattr(last_msg, "type", "") == "ai" and last_msg.content:
                            await websocket.send_json(
                                {"type": "token", "content": last_msg.content}
                            )

                # Turn complete — signal the client
                await websocket.send_json({"type": "end"})
                logger.info(
                    logger.info(
                        "Turn complete | customer_id=%s | thread_id=%s",
                        customer_id,
                        thread_id,
                    )
                )

                # ── 2f. Post-turn: check if graph set escalate_to_human ──────
                try:
                    final_state = await app_state.graph.aget_state(config)
                    if final_state.values.get("escalate_to_human"):
                        async with app_state.pool.connection() as conn:
                            await conn.execute(
                                "UPDATE customer_conversations SET status = 'escalated' WHERE conversation_id = %s",
                                (thread_id,),
                            )
                        if redis_client:
                            await redis_client.publish(
                                "support:events",
                                json.dumps(
                                    {
                                        "event": "new_escalation",
                                        "conversation_id": thread_id,
                                    }
                                ),
                            )
                        logger.info(
                            logger.info(
                                "🚨 Conversation escalated to human | thread_id=%s",
                                thread_id,
                            )
                        )
                except Exception as e:
                    logger.warning("Could not check/update escalation status: %s", e)

            except WebSocketDisconnect:
                # Client disconnected mid-stream — exit cleanly
                logger.info(
                    logger.info(
                        "Client disconnected mid-stream for customer_id=%s", customer_id
                    )
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
                    break  # socket is dead
                # continue → wait for next message, socket stays open

            except Exception as e:
                logger.error("Graph streaming error: %s", e, exc_info=True)
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except Exception:
                    break  # socket is dead
                # continue → stay connected for next turn

            # Loop back to receive_json() — connection stays open ✅

    finally:
        # Publish customer offline/disconnected event to support dashboard
        if conversation_id:
            redis_client_ref = getattr(app_state, "redis", None)
            if not redis_client_ref:
                # Fallback: use a fresh redis connection from pubsub_redis
                redis_client_ref = pubsub_redis
            try:
                await redis_client_ref.delete(f"presence:{conversation_id}")
                await redis_client_ref.publish(
                    "support:events",
                    json.dumps(
                        {
                            "event": "customer_offline",
                            "conversation_id": conversation_id,
                        }
                    ),
                )
                # Also keep legacy event name for backward compat
                await redis_client_ref.publish(
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

        # ── Cleanup Pub/Sub on disconnect ─────────────────────────────────────
        if listener_task:
            listener_task.cancel()
        if pubsub:
            await pubsub.unsubscribe()
            await pubsub.aclose()
        await pubsub_redis.aclose()


# ══════════════════════════════════════════════════════════════════════════════
# HUMAN SUPPORT AGENT API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

import pathlib

from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel as PydanticBaseModel


class SupportReplyRequest(PydanticBaseModel):
    """Model representing a support reply request. Includes fields for ticket ID, reply content, and optional attachments."""

    message: str
    agent_name: str


class TypingRequest(PydanticBaseModel):
    """Pydantic model representing a typing request, containing the fields required to submit a typing operation. Includes validation to ensure request data conforms to expected formats."""

    agent_name: str


@app.get("/support/stream")
async def support_stream(request: Request):
    """SSE endpoint for the Support Dashboard to receive real-time updates."""
    redis_client = getattr(request.app.state, "redis", None)
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not configured")

    shutdown_event: asyncio.Event = getattr(
        request.app.state, "shutdown_event", asyncio.Event()
    )

    # Use a dedicated Redis connection for pub/sub (never share with the pool)
    pubsub_redis = async_redis.Redis.from_url(os.getenv("REDIS_URL"))
    pubsub = pubsub_redis.pubsub()
    await pubsub.subscribe("support:events")

    # Feed messages into a queue so the generator is always cancellable
    queue: asyncio.Queue = asyncio.Queue()

    async def _reader():
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
        try:
            while True:
                # Race between a queued message and the server shutdown event
                get_task = asyncio.ensure_future(queue.get())
                shutdown_task = asyncio.ensure_future(shutdown_event.wait())
                done, pending = await asyncio.wait(
                    [get_task, shutdown_task],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=30,  # also send keep-alive every 30s
                )

                # Cancel the loser
                for t in pending:
                    t.cancel()

                # If the server is shutting down, exit cleanly right now
                if shutdown_event.is_set():
                    break

                # If the client disconnected, exit
                if await request.is_disconnected():
                    break

                if get_task in done:
                    data = get_task.result()
                    if data:
                        yield f"data: {data}\n\n"
                    else:
                        yield ": keep-alive\n\n"
                else:
                    # timeout — just send a keep-alive
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            reader_task.cancel()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/support/conversations")
async def list_escalated_conversations(request: Request):
    """List all conversations with status='escalated' for the support dashboard."""
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


@app.get("/support/conversations/{conversation_id}/history")
async def get_support_conversation_history(conversation_id: str, request: Request):
    """Fetch full message history for the support agent dashboard."""
    graph = request.app.state.graph
    try:
        config = {"configurable": {"thread_id": conversation_id}}
        state = await graph.aget_state(config)
        messages = state.values.get("messages", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch state: {e}")

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
                text_parts = []
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content_str = " ".join(text_parts)
            else:
                content_str = str(msg.content)

        formatted.append(
            {
                "role": role,
                "content": content_str,
                "name": name,
            }
        )

    redis_client = getattr(request.app.state, "redis", None)
    is_online = False
    if redis_client:
        try:
            val = await redis_client.get(f"presence:{conversation_id}")
            logger.info("Presence check for %s: val=%s", conversation_id, val)
            if val == b"online" or val == "online":
                is_online = True
        except Exception:
            pass

    return {"messages": formatted, "is_online": is_online}


@app.post("/support/conversations/{conversation_id}/join")
async def join_conversation(conversation_id: str, agent_name: str, request: Request):
    """Claim an escalated conversation and notify the customer."""
    pool = request.app.state.pool
    redis_client = request.app.state.redis

    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE customer_conversations SET assigned_agent = %s, updated_at = NOW() WHERE conversation_id = %s",
            (agent_name, conversation_id),
        )
        await conn.commit()

    # Publish join notification to the customer via Redis Pub/Sub
    notification = json.dumps(
        {
            "event": "agent_joined",
            "content": f"🟢 **{agent_name} (Support Agent) has joined the chat.**",
        }
    )
    await redis_client.publish(f"chat:reply:{conversation_id}", notification)
    await redis_client.publish(
        "support:events",
        json.dumps({"event": "agent_joined", "conversation_id": conversation_id}),
    )
    logger.info("Agent %s joined conversation %s", agent_name, conversation_id)

    return {"status": "joined", "agent_name": agent_name}


@app.post("/support/conversations/{conversation_id}/reply")
@observe(name="human_agent_reply", as_type="generation")
async def reply_as_human(
    conversation_id: str, body: SupportReplyRequest, request: Request
):
    """Send a reply as a human support agent."""
    graph = request.app.state.graph
    redis_client = request.app.state.redis

    # 1. Write the reply to LangGraph checkpoint history
    config = {"configurable": {"thread_id": conversation_id}}
    try:
        await graph.aupdate_state(
            config,
            {"messages": [AIMessage(content=body.message, name="support_agent")]},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save reply: {e}")

    # 1.5 Update database timestamp
    pool = request.app.state.pool
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE customer_conversations SET updated_at = NOW() WHERE conversation_id = %s",
            (conversation_id,),
        )
        await conn.commit()

    # 2. Publish the reply to the customer via Redis Pub/Sub
    notification = json.dumps(
        {
            "event": "agent_reply",
            "content": body.message,
            "agent_name": body.agent_name,
        }
    )
    await redis_client.publish(f"chat:reply:{conversation_id}", notification)
    await redis_client.publish(
        "support:events",
        json.dumps({"event": "agent_reply", "conversation_id": conversation_id}),
    )
    logger.info("Support agent replied to %s", conversation_id)

    return {"status": "sent"}


@app.post("/support/conversations/{conversation_id}/typing")
async def agent_typing(conversation_id: str, req: TypingRequest, request: Request):
    """Broadcast to customer that the support agent is typing."""
    pool = getattr(request.app.state, "pool", None)
    if pool:
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE customer_conversations SET updated_at = NOW() WHERE conversation_id = %s",
                (conversation_id,),
            )
            await conn.commit()

    redis_client = getattr(request.app.state, "redis", None)
    if not redis_client:
        return {"status": "ignored"}

    notification = json.dumps({"event": "agent_typing", "agent_name": req.agent_name})
    await redis_client.publish(f"chat:reply:{conversation_id}", notification)
    return {"status": "sent"}


@app.post("/chat/conversations/{conversation_id}/typing/customer")
async def customer_typing(conversation_id: str, request: Request):
    """Broadcast to support agent dashboard that the customer is typing."""
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client:
        await redis_client.publish(
            "support:events",
            json.dumps(
                {"event": "customer_typing", "conversation_id": conversation_id}
            ),
        )
    return {"status": "sent"}


@app.post("/support/conversations/{conversation_id}/resolve")
async def resolve_conversation(conversation_id: str, request: Request):
    """Resolve an escalated conversation and return it to AI handling."""
    pool = request.app.state.pool
    graph = request.app.state.graph
    redis_client = request.app.state.redis

    # 1. Update database status back to 'active'
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE customer_conversations SET status = 'active', assigned_agent = NULL WHERE conversation_id = %s",
            (conversation_id,),
        )
        await conn.commit()

    # 2. Reset escalation flag and clear active issues in LangGraph state
    config = {"configurable": {"thread_id": conversation_id}}
    try:
        state = await graph.aget_state(config)
        chat_summary = state.values.get("chat_summary", "")

        if chat_summary and "### Active Issues" in chat_summary:
            import re

            active_block_match = re.search(
                r"### Active Issues\n(.*?)(?=\n### )", chat_summary, re.DOTALL
            )
            resolved_block_match = re.search(
                r"### Resolved Issues\n(.*?)(?=\n### )", chat_summary, re.DOTALL
            )

            if active_block_match and resolved_block_match:
                active_text = active_block_match.group(1).strip()
                resolved_text = resolved_block_match.group(1).strip()

                if active_text and active_text != "- None":
                    if resolved_text == "- None":
                        resolved_text = ""
                    cleaned_active = re.sub(
                        r"- \[Turns Active: \d+\] ", "- ", active_text
                    )
                    new_resolved = (resolved_text + "\n" + cleaned_active).strip()
                    chat_summary = (
                        chat_summary[: resolved_block_match.start(1)]
                        + new_resolved
                        + "\n"
                        + chat_summary[resolved_block_match.end(1) :]
                    )

                chat_summary = (
                    chat_summary[: active_block_match.start(1)]
                    + "- None\n"
                    + chat_summary[active_block_match.end(1) :]
                )

            chat_summary = re.sub(
                r"### Escalate to Human\n- True",
                "### Escalate to Human\n- False",
                chat_summary,
            )

        await graph.aupdate_state(
            config,
            {
                "escalate_to_human": False,
                "chat_summary": chat_summary,
                "messages": [
                    SystemMessage(
                        content="SYSTEM: The human support agent has resolved this issue. Automated AI assistant is now re-enabled."
                    )
                ],
            },
        )
    except Exception as e:
        logger.error("Failed to reset escalation state: %s", e)

    # 3. Notify customer via Redis Pub/Sub
    notification = json.dumps(
        {
            "event": "resolved",
            "content": "✅ *The support agent has resolved this issue. The AI assistant is ready to help you with other questions!*",
        }
    )
    await redis_client.publish(f"chat:reply:{conversation_id}", notification)
    await redis_client.publish(
        "support:events",
        json.dumps({"event": "resolved", "conversation_id": conversation_id}),
    )
    logger.info("Conversation %s resolved and returned to AI", conversation_id)

    return {"status": "resolved"}


@app.get("/support/dashboard", response_class=HTMLResponse)
async def support_dashboard():
    """Serve the human support agent dashboard HTML page."""
    template_path = (
        pathlib.Path(__file__).parent / "templates" / "support_dashboard.html"
    )
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard template not found")
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
