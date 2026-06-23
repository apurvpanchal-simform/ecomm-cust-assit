"""
FastAPI application lifespan manager.
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
import redis.asyncio as async_redis
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.db.migrations import setup_postgres_tables
from app.graph.builder import compile_graph
from app.graph.checkpointer import AsyncDualCheckpointer
from app.tasks.agent_sweep import _agent_inactivity_sweep

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for initializing all global infrastructure.

    Sets up (in order):
    1. Postgres connection pool + required table migrations.
    2. Redis connection + agent inactivity sweep background task.
    3. LangChain global LLM cache (Redis exact-match + image bypass).
    4. AsyncDualCheckpointer (Redis + Postgres).
    5. LangGraph compiled application graph.

    Teardown (on shutdown):
    - Signals the inactivity sweep to stop.
    - Cancels the sweep background task.

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
            # ── 1. Postgres table migrations ──────────────────────────────────
            logger.info("Initializing AsyncPostgresSaver...")
            postgres_saver = AsyncPostgresSaver(pool)
            await postgres_saver.setup()
            await setup_postgres_tables(pool)

            # ── 2. Background sweep task ───────────────────────────────────────
            app.state.pool = pool
            app.state.redis = redis_client
            app.state.shutdown_event = asyncio.Event()
            sweep_task = asyncio.create_task(_agent_inactivity_sweep(app))

            # ── 4. Dual checkpointer ──────────────────────────────────────────
            logger.info("Initializing AsyncRedisSaver...")
            redis_saver = AsyncRedisSaver(redis_client=redis_client)
            await redis_saver.asetup()

            dual_checkpointer = AsyncDualCheckpointer(
                redis_saver=redis_saver,
                postgres_saver=postgres_saver,
                pool=pool,
            )

            # ── 5. Compile the LangGraph ──────────────────────────────────────
            app.state.graph = compile_graph(checkpointer=dual_checkpointer)
            logger.info(
                "Graph compiled successfully with Redis + Postgres async checkpointing."
            )

            yield  # Application is running

            # ── Shutdown ──────────────────────────────────────────────────────
            # Signal long-lived SSE streams to exit before Uvicorn waits on connections
            app.state.shutdown_event.set()
            app.state.is_shutting_down = True
            sweep_task.cancel()

    except Exception:
        logger.exception("Failed to initialize graph infrastructure.")
        raise
