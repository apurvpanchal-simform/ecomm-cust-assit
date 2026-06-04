import os
import redis
from contextlib import asynccontextmanager
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from dotenv import load_dotenv

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.postgres import PostgresSaver
from app.graph.builder import compile_graph
from app.graph.checkpointer import DualCheckpointer

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi import HTTPException
from app.schemas.auth import (LoginRequest,LoginResponse,)
from app.schemas.input import (ChatRequest,)
from app.services.auth_service import (generate_jwt,)
from app.services.customer_service import (CustomerService,)
from fastapi import Depends
from app.middleware.auth import (get_current_customer)

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.getenv("REDIS_URL")
    supabase_db_url = os.getenv("SUPABASE_DB_URL")
    
    if not redis_url or not supabase_db_url:
        print("Error: Missing REDIS_URL / SUPABASE_DB_URL")
        app.state.graph = compile_graph() # fallback
        yield
        return

    with ConnectionPool(
        supabase_db_url,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}
    ) as pool, redis.Redis.from_url(redis_url) as redis_client:
        postgres_saver = PostgresSaver(pool)
        postgres_saver.setup()
        
        redis_saver = RedisSaver(redis_client=redis_client)
        redis_saver.setup()
        dual_checkpointer = DualCheckpointer(
            redis_saver=redis_saver, 
            postgres_saver=postgres_saver
        )
        app.state.graph = compile_graph(checkpointer=dual_checkpointer)
        print("Graph compiled successfully with Redis & Supabase checkpointing.")
        yield

app = FastAPI(lifespan=lifespan)

@app.post(
    "/auth/login",
    response_model=LoginResponse
)
async def login(
    request: LoginRequest
):

    customer_service = CustomerService()

    customer = (
        customer_service.get_customer_by_email(
            request.email
        )
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

    config = {"configurable": {"thread_id": customer_id}}

    result = request.app.state.graph.invoke(
        {
            "query": chat_request.query,
            "customer_id": customer_id,
        },
        config=config
    )

    return result