import logging
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)


async def setup_postgres_tables(pool: AsyncConnectionPool) -> None:
    """
    Create or migrate all required Postgres tables at startup.

    Creates the `checkpoint_state_logs` audit table and its index, and ensures
    the `customer_conversations` table has the required escalation columns.

    Args:
        pool: Async Postgres connection pool.
    """
    async with pool.connection() as conn:
        # Audit log table for human-readable state inspection
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

    async with pool.connection() as conn:
        # Add escalation columns if they don't already exist (idempotent ALTER)
        await conn.execute("ALTER TABLE customer_conversations ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active';")
        await conn.execute("ALTER TABLE customer_conversations ADD COLUMN IF NOT EXISTS chat_summary TEXT;")
        await conn.execute("ALTER TABLE customer_conversations ADD COLUMN IF NOT EXISTS assigned_agent TEXT;")
    logger.info("Ensured escalation columns (status, assigned_agent) exist.")
