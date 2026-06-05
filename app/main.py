import os
import logging
import redis

from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException, Depends

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.postgres import PostgresSaver

from app.graph.builder import compile_graph
from app.graph.checkpointer import DualCheckpointer

from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
)

from app.schemas.input import (
    ChatRequest,
)

from pydantic import BaseModel
from typing import List
from datetime import datetime

class ConversationItem(BaseModel):
    conversation_id: str
    title: str
    updated_at: datetime

from app.services.auth_service import (
    generate_jwt,
)

from app.services.customer_service import (
    CustomerService,
)

from app.middleware.auth import (
    get_current_customer,
)

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    redis_url = os.getenv("REDIS_URL")
    supabase_db_url = os.getenv("SUPABASE_DB_URL")

    if not redis_url:
        raise RuntimeError(
            "REDIS_URL environment variable is missing"
        )

    if not supabase_db_url:
        raise RuntimeError(
            "SUPABASE_DB_URL environment variable is missing"
        )

    try:

        with (
            ConnectionPool(
                supabase_db_url,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                },
            ) as pool,
            redis.Redis.from_url(redis_url) as redis_client,
        ):

            print("Initializing PostgresSaver...")
            postgres_saver = PostgresSaver(pool)
            postgres_saver.setup()

            print("Ensuring customer_conversations table exists...")
            with pool.connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS customer_conversations (
                        conversation_id TEXT PRIMARY KEY,
                        customer_id TEXT NOT NULL,
                        title TEXT,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
            
            app.state.pool = pool

            print("Initializing RedisSaver...")
            redis_saver = RedisSaver(
                redis_client=redis_client
            )
            redis_saver.setup()

            dual_checkpointer = DualCheckpointer(
                redis_saver=redis_saver,
                postgres_saver=postgres_saver,
            )

            app.state.graph = compile_graph(
                checkpointer=dual_checkpointer
            )

            print(
                "Graph compiled successfully with Redis + Postgres checkpointing."
            )

            yield

    except Exception:
        logger.exception(
            "Failed to initialize graph infrastructure."
        )
        raise


app = FastAPI(lifespan=lifespan)


@app.post(
    "/auth/login",
    response_model=LoginResponse,
)
async def login(
    request: LoginRequest,
):

    customer_service = CustomerService()

    customer = customer_service.get_customer_by_email(
        request.email
    )

    if not customer:
        raise HTTPException(
            status_code=401,
            detail="Customer not found",
        )

    token = generate_jwt(
        customer_id=customer["customer_id"],
        email=customer["email"],
    )

    return LoginResponse(
        access_token=token
    )


@app.post("/chat")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    customer_id: str = Depends(
        get_current_customer
    ),
):

    thread_id = (
        chat_request.conversation_id
        or customer_id
    )

    with request.app.state.pool.connection() as conn:
        conn.execute("""
            INSERT INTO customer_conversations (conversation_id, customer_id, title)
            VALUES (%s, %s, %s)
            ON CONFLICT (conversation_id) DO UPDATE SET updated_at = NOW()
        """, (thread_id, customer_id, chat_request.query[:30] + "..."))

    print(
        f"Invoking graph | "
        f"customer_id={customer_id} | "
        f"thread_id={thread_id}"
    )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = request.app.state.graph.invoke(
        {
            "query": chat_request.query,
            "customer_id": customer_id,
        },
        config=config,
    )

    return result


@app.get("/chat/conversations", response_model=List[ConversationItem])
async def list_conversations(
    request: Request,
    customer_id: str = Depends(get_current_customer),
):
    with request.app.state.pool.connection() as conn:
        result = conn.execute("""
            SELECT conversation_id, title, updated_at 
            FROM customer_conversations 
            WHERE customer_id = %s 
            ORDER BY updated_at DESC
        """, (customer_id,)).fetchall()
        
        return [ConversationItem(**row) for row in result]


@app.get("/chat/history/{conversation_id}")
async def get_chat_history(
    conversation_id: str,
    request: Request,
    customer_id: str = Depends(get_current_customer),
):
    with request.app.state.pool.connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM customer_conversations WHERE conversation_id = %s AND customer_id = %s",
            (conversation_id, customer_id)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=403, detail="Conversation not found")

    config = {"configurable": {"thread_id": conversation_id}}
    state = request.app.state.graph.get_state(config)
    
    messages = state.values.get("messages", [])
    
    formatted_messages = []
    for msg in messages:
        # Avoid including empty or pure tool-call messages
        if msg.type in ["human", "ai"] and msg.content:
            formatted_messages.append({
                "role": "user" if msg.type == "human" else "assistant",
                "content": str(msg.content)
            })
            
    return {"messages": formatted_messages}