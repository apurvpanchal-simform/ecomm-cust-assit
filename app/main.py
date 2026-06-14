"""
Main FastAPI application entry point for the E-Commerce Customer Assistant.

Sets up the LangGraph pipeline, authentication, and chat endpoints.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import List

import redis
import redis.asyncio as async_redis
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Query
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.cache import RedisCache
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
from app.middleware.auth import (
    get_current_customer,
)
from app.middleware.rate_limit import (
    rate_limit_customer,
)
from app.schemas.api import (
    ConversationItem,
    LoginRequest,
    LoginResponse,
)
from app.services.jwt_auth import (
    generate_jwt,
    verify_jwt,
)
from app.services.faq_response_cache import get_faq_response

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

            app.state.pool = pool
            app.state.redis = redis_client

            logger.info("Initializing AsyncRedisSaver...")
            redis_saver = AsyncRedisSaver(redis_client=redis_client)
            await redis_saver.asetup()

            class LoggingRedisCache(RedisCache):
                def lookup(self, prompt: str, llm_string: str):
                    hit = super().lookup(prompt, llm_string)
                    if hit:
                        logger.info("🟢 CACHE HIT! Returning cached response.")
                    else:
                        logger.info("🔴 CACHE MISS! Generating new response...")
                    return hit

                async def alookup(self, prompt: str, llm_string: str):
                    hit = await super().alookup(prompt, llm_string)
                    if hit:
                        logger.info("🟢 CACHE HIT! Returning cached response.")
                    else:
                        logger.info("🔴 CACHE MISS! Generating new response...")
                    return hit

            logger.info("Enabling global LangChain LLM Redis EXACT MATCH cache with 1-hour TTL...")
            sync_redis_client = redis.Redis.from_url(redis_url)
            set_llm_cache(
                LoggingRedisCache(
                    redis_=sync_redis_client,
                    ttl=3600
                )
            )

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

    except Exception:
        logger.exception("Failed to initialize graph infrastructure.")
        raise


app = FastAPI(lifespan=lifespan)


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
            "SELECT 1 FROM customer_conversations WHERE conversation_id = %s AND customer_id = %s",
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

    return {"messages": formatted_messages}


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

@app.websocket("/chat/ws")
async def chat_ws(websocket: WebSocket, token: str = Query(...)):
    """
    WebSocket endpoint for real-time streaming of LangGraph agent events.
    Maintains a persistent connection across multiple conversation turns.
    """
    # ── 1. Authenticate ──────────────────────────────────────────────────
    try:
        payload = verify_jwt(token)
        customer_id = payload["sub"]
    except Exception as e:
        logger.error(f"JWT verification failed: {e}")
        await websocket.close(code=1008, reason="Invalid token")
        return

    await websocket.accept()
    logger.info(f"WebSocket accepted for customer_id={customer_id}")

    app_state = websocket.scope["app"].state

    # ── 2. Message loop — one iteration per user turn ────────────────────
    while True:
        # ── 2a. Receive ──────────────────────────────────────────────────
        try:
            data = await websocket.receive_json()
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected (receive) for customer_id={customer_id}")
            return
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return  # unrecoverable — socket is broken

        query: str = data.get("query", "")
        thread_id: str = data.get("conversation_id") or customer_id
        image_base64: str | None = data.get("image_base64")

        if not query:
            await websocket.send_json({"type": "error", "message": "Empty query."})
            continue  # stay connected, wait for next message

        # ── 2b. Rate Limiting ───────────────────────────────────────────
        redis_client = getattr(app_state, "redis", None)
        if redis_client:
            key = f"rate_limit:{customer_id}"
            async with redis_client.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.expire(key, 60, nx=True)
                results = await pipe.execute()
            
            if results[0] > 10:
                await websocket.send_json({
                    "type": "error", 
                    "message": "Too Many Requests. Please wait a minute before trying again."
                })
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
                    (thread_id, customer_id, (query[:30] + "...") if len(query) > 30 else query),
                )
        except Exception as e:
            logger.error(f"DB error saving conversation: {e}")
            await websocket.send_json({"type": "error", "message": "Failed to save conversation."})
            continue  # not fatal — still try to run the graph

        # --- Application-level Response Cache Hit Check (Bypass Graph) ---
        cached_response = get_faq_response(customer_id, query)
        if cached_response:
            logger.info(f"🟢 FAQ RESPONSE CACHE HIT (Bypass Graph) | customer={customer_id} | query='{query}'")
            try:
                config = RunnableConfig(configurable={"thread_id": thread_id})
                await app_state.graph.aupdate_state(
                    config,
                    {
                        "messages": [
                            HumanMessage(content=query),
                            AIMessage(content=cached_response, name="faq")
                        ],
                        "customer_id": customer_id,
                        "query": query,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to update graph state for cached FAQ response: {e}")

            await websocket.send_json({"type": "token", "content": cached_response})
            await websocket.send_json({"type": "end"})
            continue

        logger.info(f"WS graph invoke | customer_id={customer_id} | thread_id={thread_id}")

        # ── 2c. Build graph config ───────────────────────────────────────
        langfuse_handler = CallbackHandler()
        config = RunnableConfig(
            configurable={"thread_id": thread_id},
            callbacks=[langfuse_handler],
        )

        # ── 2d. Crash recovery ───────────────────────────────────────────
        # Build state_input first; may be set to None if we are resuming a crash.
        state_input = {
            "query": query,
            "customer_id": customer_id,
            "messages": [HumanMessage(content=query)],
            "image_base64": image_base64 or None,
        }

        try:
            current_state = await app_state.graph.aget_state(config)
            if current_state.next:
                logger.info(
                    f"Crash recovery: pending nodes={current_state.next}. "
                    f"Injecting fresh query and resuming."
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
            logger.warning(f"Could not check/resume graph state: {e}. Starting fresh.")

        # ── 2e. Stream graph events ──────────────────────────────────────
        # This is isolated in its own try/except so that ANY error during
        # streaming sends {"type":"error"} and then CONTINUES the while loop
        # instead of propagating up to the outer scope and closing the socket.
        try:
            tokens_streamed = False
            async for event in app_state.graph.astream_events(
                state_input,
                config=config,
                version="v2",
            ):
                event_type = event["event"]

                if event_type == "on_chat_model_stream":
                    metadata = event.get("metadata", {})
                    node = metadata.get("langgraph_node")
                    if node not in ["faq", "order", "synthesizer"]:
                        continue

                    chunk = event["data"]["chunk"]
                    # chunk.content can be a string or a list of content blocks
                    content = chunk.content
                    if isinstance(content, list):
                        # Extract text from content blocks (e.g. Anthropic format)
                        content = "".join(
                            block.get("text", "") if isinstance(block, dict) else str(block)
                            for block in content
                        )
                    if content:
                        tokens_streamed = True
                        await websocket.send_json({"type": "token", "content": content})

                elif event_type == "on_tool_start":
                    await websocket.send_json({
                        "type": "tool_start",
                        "name": event["name"],
                        "inputs": str(event["data"].get("input", {})),
                    })

                elif event_type == "on_tool_end":
                    await websocket.send_json({
                        "type": "tool_end",
                        "name": event["name"],
                        "output": str(event["data"].get("output", "")),
                    })

            # If no tokens were streamed (e.g. cache hit or direct supervisor response),
            # send the last AI message from the final graph state.
            if not tokens_streamed:
                current_state = await app_state.graph.aget_state(config)
                messages = current_state.values.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if getattr(last_msg, "type", "") == "ai" and last_msg.content:
                        await websocket.send_json({"type": "token", "content": last_msg.content})

            # Turn complete — signal the client
            await websocket.send_json({"type": "end"})
            logger.info(f"Turn complete | customer_id={customer_id} | thread_id={thread_id}")

        except WebSocketDisconnect:
            # Client disconnected mid-stream — exit cleanly
            logger.info(f"Client disconnected mid-stream for customer_id={customer_id}")
            return

        except asyncio.TimeoutError:
            logger.error("Graph execution timed out.")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": "Request timed out. Please try again.",
                })
            except Exception:
                return  # socket is dead
            # continue → wait for next message, socket stays open

        except Exception as e:
            logger.error(f"Graph streaming error: {e}", exc_info=True)
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                return  # socket is dead
            # continue → stay connected for next turn

        # Loop back to receive_json() — connection stays open ✅