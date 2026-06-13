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
from fastapi import Depends, FastAPI, HTTPException, Request
from langchain_core.globals import set_llm_cache
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.cache import RedisCache
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
from app.middleware.auth import (
    get_current_customer,
)
from app.middleware.rate_limit import (
    rate_limit_customer,
)
from app.schemas.api import (
    ChatRequest,
    ConversationItem,
    LoginRequest,
    LoginResponse,
)
from app.services.faq_response_cache import get_faq_response
from app.services.jwt_auth import (
    generate_jwt,
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


@app.post("/chat")
@observe()
async def chat(
    request: Request,
    chat_request: ChatRequest,
    customer_id: str = Depends(rate_limit_customer),
):
    """
    Main chat endpoint to interact with the LangGraph agent.

    Processes user messages, manages conversation threads, handles checkpoints,
    and returns the final agent state. Reconnects to previous state if a crash occurred.

    Args:
        request: The FastAPI request object.
        chat_request: The chat payload containing query and conversation metadata.
        customer_id: The authenticated customer ID (injected via dependency).

    Returns:
        The final state of the conversation after executing the graph.

    Raises:
        HTTPException: On timeout or conversation length limits.
    """
    from langfuse import propagate_attributes

    thread_id = chat_request.conversation_id or customer_id

    with propagate_attributes(session_id=thread_id, user_id=customer_id):
        async with request.app.state.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO customer_conversations (conversation_id, customer_id, title)
                VALUES (%s, %s, %s)
                ON CONFLICT (conversation_id) DO UPDATE SET updated_at = NOW()
            """,
                (thread_id, customer_id, chat_request.query[:30] + "..."),
            )

        logger.info(
            f"Invoking graph | customer_id={customer_id} | thread_id={thread_id}"
        )

        # ── API-Level FAQ Cache Pre-Check ─────────────────────────────────────
        # Check the FAQ response cache BEFORE invoking the graph so that
        # cached FAQ hits completely bypass the supervisor, all LangGraph
        # checkpointing, and any LLM calls.
        if not chat_request.image_base64:  # Never short-circuit image queries
            faq_hit = get_faq_response(customer_id, chat_request.query)
            if faq_hit:
                logger.info(
                    f"⚡ API-LEVEL FAQ CACHE HIT — skipping graph entirely for query='{chat_request.query}'"
                )
                return {
                    "messages": [
                        {"type": "ai", "content": faq_hit, "name": "faq"}
                    ],
                    "query": chat_request.query,
                    "customer_id": customer_id,
                    "error": None,
                    "faq_chunks": [],
                }

        langfuse_handler = CallbackHandler()
        config = RunnableConfig(
            configurable={"thread_id": thread_id},
            callbacks=[langfuse_handler],
        )
        # Enforce conversation length limit
        current_state = await request.app.state.graph.aget_state(config)
        if len(current_state.values.get("messages", [])) > 100:
            raise HTTPException(
                status_code=400,
                detail="Conversation has reached its maximum length. Please start a new chat.",
            )

        state_input = {
            "query": chat_request.query,
            "customer_id": customer_id,
            "messages": [HumanMessage(content=chat_request.query)],
            "image_base64": chat_request.image_base64 or None,
        }

        # ── Crash Recovery ────────────────────────────────────────────────────
        # If the graph crashed previously it will have pending 'next' nodes in
        # the checkpoint. Two scenarios:
        #
        # 1. Same query as the crashed one → user is retrying. Resume the graph
        #    from exactly where it left off (don't inject a duplicate HumanMessage).
        #
        # 2. Different query → user has moved on to a new question. Clear the
        #    stale pending state and process the new query from scratch so that
        #    the old crashed turn doesn't silently hijack the new request.
        # ── Continuation intent keywords ───────────────────────────────────
        # Short messages that signal "please resume / retry" rather than a
        # new question. Checked only when a crash is already pending.
        _CONTINUATION_SIGNALS = frozenset({
            "go ahead", "continue", "proceed", "retry", "try again",
            "what happened", "what happened to my query", "what happened to my above query",
            "what happened to my request", "what happened to my above request", "finish", "resume",
            "go on", "please continue", "please proceed", "what about my above query", "what about my query",
            "what about my request", "what about my above request", "any update", "still processing",
            "are you there", "hello", "hi", "hey",
        })

        def _is_continuation_intent(msg: str) -> bool:
            """Return True if msg looks like a resume/retry signal."""
            normalized = msg.strip().lower().rstrip("?!.")
            
            # 1. Exact match against known short phrases
            if normalized in _CONTINUATION_SIGNALS:
                return True
                
            # 2. Substring match for longer descriptive continuation phrases
            long_signals = [
                "previous request", "last request", "previous query", "last query", 
                "interrupted", "crashed", "status of", "my answer", "earlier query"
            ]
            if any(signal in normalized for signal in long_signals):
                return True

            # 3. Short message heuristic (≤4 words)
            # Short messages like "yes please" or "go" are often continuations.
            # But if they contain question words or domain keywords, they are new queries.
            words = normalized.split()
            if len(words) <= 4:
                domain_and_question_words = {
                    "what", "where", "how", "when", "who", "why", "is", "are", 
                    "do", "does", "can", "could", "price", "cost", "track", 
                    "cancel", "return", "refund", "shipping", "delivery", "late", 
                    "order", "policy", "buy", "purchase", "item", "product"
                }
                # If none of the words are questions/domain keywords, treat as continuation
                if not any(w in domain_and_question_words for w in words):
                    return True
                    
            return False

        if current_state.next:
            crashed_query = current_state.values.get("query", "")
            is_retry = crashed_query.strip().lower() == chat_request.query.strip().lower()
            is_continuation = _is_continuation_intent(chat_request.query)

            if is_retry or is_continuation:
                reason = "same query" if is_retry else f"continuation signal ('{chat_request.query}')"
                logger.info(
                    f"Resuming crashed execution ({reason}). Pending nodes: {current_state.next}"
                )
                # Only refresh session identity — do NOT add a duplicate HumanMessage.
                resume_state = {
                    "customer_id": customer_id,
                    "query": crashed_query,
                }
                await request.app.state.graph.aupdate_state(config, resume_state)
                state_input = None
            else:
                logger.info(
                    f"User moved on to a new query. Discarding stale crash state "
                    f"(was: '{crashed_query}', now: '{chat_request.query}'). "
                    f"Starting fresh."
                )
                # state_input is already set correctly for the new query above —
                # pass it as-is so the graph restarts from the supervisor.

        from langchain_core.globals import get_llm_cache
        logger.info(f"--- [DEBUG] GLOBAL CACHE SETTING: {get_llm_cache()} ---")

        logger.info("--- GRAPH EXECUTION START ---")

        async def consume_graph():
            async for event in request.app.state.graph.astream(
                state_input,
                config=config,
                stream_mode="updates",
            ):
                for node_name, state_update in event.items():
                    logger.info(f"--- [NODE INVOKED]: {node_name} ---")

                    if state_update is not None:
                        if "messages" in state_update:
                            msgs = state_update["messages"]
                            if not isinstance(msgs, list):
                                msgs = [msgs]
                            for msg in msgs:
                                if getattr(msg, "tool_calls", None):
                                    logger.info(f"🛠️ [TOOL CALLS]: {msg.tool_calls}")
                                if getattr(msg, "content", None):
                                    logger.info(f"💬 [MESSAGE CONTENT]: {msg.content}")

                        safe_update = {
                            k: v
                            for k, v in state_update.items()
                            if k not in ["messages", "image_base64", "image_embedding"]
                        }
                        if safe_update:
                            try:
                                logger.info(
                                    f"🔄 [STATE UPDATE]: {json.dumps(safe_update, default=str)}"
                                )
                            except Exception:
                                logger.info(f"🔄 [STATE UPDATE]: {safe_update}")

        try:
            await asyncio.wait_for(consume_graph(), timeout=90.0)
        except asyncio.TimeoutError:
            logger.error("Graph execution timed out after 90 seconds.")
            raise HTTPException(
                status_code=504,
                detail="Request to the agent timed out. Please try again.",
            )

        final_state = await request.app.state.graph.aget_state(config)
        logger.info("--- GRAPH EXECUTION END ---")

        result = final_state.values
        safe_result = {
            k: v
            for k, v in result.items()
            if k not in ["messages", "image_base64", "image_embedding"]
        }
        logger.info(f"🏁 [FINAL AGENT STATE]: {json.dumps(safe_result, default=str)}")

        return result


@app.get("/chat/conversations", response_model=List[ConversationItem])
async def list_conversations(
    request: Request,
    customer_id: str = Depends(get_current_customer),
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
    customer_id: str = Depends(get_current_customer),
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
                formatted_messages.append(
                    {
                        "role": "user",
                        "content": str(msg.content),
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
    customer_id: str = Depends(get_current_customer),
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
    customer_id: str = Depends(get_current_customer),
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
    customer_id: str = Depends(get_current_customer),
):
    """Fetch raw products from Supabase."""
    supabase = await get_supabase_client()
    result = await supabase.table("products").select("*").execute()
    return result.data
