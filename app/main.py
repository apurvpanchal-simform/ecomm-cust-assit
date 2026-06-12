from langchain_core.runnables import RunnableConfig
import os
import asyncio
import logging
import json
import redis.asyncio as redis

from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException, Depends
from langchain_core.messages import HumanMessage

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.graph.builder import compile_graph
from app.graph.checkpointer import AsyncDualCheckpointer

from app.schemas.api import (
    LoginRequest,
    LoginResponse,
    ChatRequest,
    ConversationItem,
)

from typing import List
from app.services.jwt_auth import (
    generate_jwt,
)

from app.db.customers import (
    CustomerService,
)

from app.middleware.auth import (
    get_current_customer,
)

from app.middleware.rate_limit import (
    rate_limit_customer,
)

from app.db.supabase import get_supabase_client

from langfuse import observe
from langfuse.langchain import CallbackHandler

load_dotenv(override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):

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
            redis.Redis.from_url(redis_url) as redis_client,
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
    return {"status": "ok"}


@app.post(
    "/auth/login",
    response_model=LoginResponse,
)
async def login(
    request: LoginRequest,
):

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

    thread_id = chat_request.conversation_id or customer_id

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
        f"Invoking graph | " f"customer_id={customer_id} | " f"thread_id={thread_id}"
    )

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
        "pending_agents": [],
        "executed_agents": [],
    }

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
            status_code=504, detail="Request to the agent timed out. Please try again."
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
            synth_msg = next((m for m in current_turn_ai_messages if getattr(m, "name", "") == "synthesizer"), None)
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
