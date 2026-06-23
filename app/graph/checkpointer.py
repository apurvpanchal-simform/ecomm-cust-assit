"""
Custom LangGraph Checkpointer implementing a Dual-Write strategy.

Read path:   Redis first (fast) → Postgres fallback (durable)
Write path:  Postgres (durable) + Redis (hot cache) simultaneously

Why dual-write?
---------------
- Redis provides sub-millisecond reads for active conversations.
- Postgres (Supabase) provides durability so conversations survive Redis restarts.
- Redis keys are given a 1-hour TTL to bound memory usage.

Additionally, every `aput()` writes a human-readable JSON row to the
`checkpoint_state_logs` table in Postgres for debugging and auditing.
"""

import json
import logging
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

logger = logging.getLogger(__name__)


class AsyncDualCheckpointer(BaseCheckpointSaver):
    """
    Async LangGraph checkpointer that writes to both Redis and Postgres.

    Reads prioritise Redis for speed; on a cache miss the checkpointer falls
    back to Postgres and re-warms the Redis cache so the next read is fast.

    Attributes:
        redis_saver:    The AsyncRedisSaver for hot-path reads/writes.
        postgres_saver: The AsyncPostgresSaver for durable storage.
        pool:           Optional psycopg connection pool used to write the
                        human-readable JSON audit log.
    """

    def __init__(
        self,
        redis_saver: BaseCheckpointSaver,
        postgres_saver: BaseCheckpointSaver,
        pool: Any = None,
    ):
        super().__init__()
        self.redis_saver = redis_saver
        self.postgres_saver = postgres_saver
        self.pool = pool

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _expire_thread_keys(self, thread_id: str, ttl: int = 3600) -> None:
        """
        Set a TTL on all LangGraph checkpoint keys associated with a thread.

        Scans for keys matching `checkpoint*<thread_id>*` and calls EXPIRE on
        each one so old checkpoints are automatically evicted from Redis.

        Args:
            thread_id: The conversation thread ID.
            ttl: Expiry in seconds (default 1 hour).
        """
        if not thread_id or not hasattr(self.redis_saver, "_redis"):
            return

        try:
            redis_client = self.redis_saver._redis
            cursor = 0
            while True:
                cursor, keys = await redis_client.scan(
                    cursor=cursor, match=f"checkpoint*{thread_id}*", count=100
                )
                if keys:
                    # Set TTL for all matched keys in a single pipeline call
                    async with redis_client.pipeline(transaction=False) as pipe:
                        for key in keys:
                            pipe.expire(key, ttl)
                        await pipe.execute()
                if cursor == 0:
                    break  # Full scan complete
        except Exception as e:
            logger.error("Failed to set Redis TTL for thread %s: %s", thread_id, e)

    def _serialize_message(self, msg: Any) -> Any:
        """
        Recursively convert a LangChain message (or nested structure) to a
        JSON-serialisable dict.

        Used when writing the human-readable audit log to Postgres so that
        message objects are stored as plain dicts rather than pickled bytes.

        Args:
            msg: A LangChain message object, dict, list, or primitive value.

        Returns:
            A JSON-serialisable representation of the input.
        """
        if not msg:
            return msg

        # LangChain message objects expose a `.type` attribute
        if hasattr(msg, "type"):
            try:
                return {
                    "type": msg.type,
                    "content": getattr(msg, "content", ""),
                    "name": getattr(msg, "name", None),
                    "tool_calls": getattr(msg, "tool_calls", None),
                    "id": getattr(msg, "id", None),
                }
            except Exception:
                pass

        # Recursively handle nested dicts and lists
        if isinstance(msg, dict):
            return {k: self._serialize_message(v) for k, v in msg.items()}
        if isinstance(msg, (list, tuple)):
            return [self._serialize_message(x) for x in msg]

        return msg

    def _serialize_state_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialise the full channel values dict for the JSON audit log.

        Large binary fields (`image_base64`, `image_embedding`) are replaced
        with a placeholder string to avoid bloating the Postgres audit table.

        Args:
            values: The raw `channel_values` dict from a Checkpoint object.

        Returns:
            A JSON-serialisable dict with binary fields excluded.
        """
        if not values:
            return {}

        serialized = {}
        for k, v in values.items():
            if k in ("image_base64", "image_embedding"):
                # These can be megabytes — exclude from the readable log
                serialized[k] = "[EXCLUDED_FOR_SIZE]"
                continue
            serialized[k] = self._serialize_message(v)
        return serialized

    def _serialize_metadata(self, metadata: Any) -> Dict[str, Any]:
        """
        Normalise checkpoint metadata into a plain dict for JSON serialisation.

        Handles Pydantic models (`.dict()`), raw dicts, and anything that is
        already JSON-serialisable.  Falls back to `{"raw": str(metadata)}` if
        the object cannot be converted.

        Args:
            metadata: A CheckpointMetadata object or compatible type.

        Returns:
            A plain dict representation.
        """
        if not metadata:
            return {}
        if hasattr(metadata, "dict") and callable(metadata.dict):
            try:
                return metadata.dict()
            except Exception:
                pass
        if isinstance(metadata, dict):
            return metadata
        try:
            json.dumps(metadata)  # test if already serialisable
            return metadata
        except TypeError:
            return {"raw": str(metadata)}

    def _extract_checkpoint_fields(
        self, checkpoint: Checkpoint | None, metadata: CheckpointMetadata | None
    ) -> tuple[str, str | None, dict, str | None]:
        """
        Extract the key fields needed for the JSON audit log from raw checkpoint objects.

        Handles both Pydantic-style objects (attribute access) and plain dicts.

        Args:
            checkpoint: The LangGraph Checkpoint object.
            metadata:   The associated CheckpointMetadata object.

        Returns:
            A tuple of:
                checkpoint_id        (str)       — unique checkpoint UUID
                parent_checkpoint_id (str|None)  — parent checkpoint UUID
                channel_values       (dict)       — full state at this checkpoint
                step_node            (str|None)   — graph node name for this step
        """
        checkpoint_id = ""
        parent_checkpoint_id = None
        channel_values: dict = {}
        step_node = None

        # ── Extract from checkpoint ───────────────────────────────────────────
        if checkpoint:
            if isinstance(checkpoint, dict):
                channel_values = checkpoint.get("channel_values", {})
                checkpoint_id = checkpoint.get("id", "")
                parent_checkpoint_id = checkpoint.get("parent_checkpoint_id")
            else:
                channel_values = getattr(checkpoint, "channel_values", {})
                checkpoint_id = getattr(checkpoint, "id", "")
                parent_checkpoint_id = getattr(checkpoint, "parent_checkpoint_id", None)

        # ── Extract from metadata ─────────────────────────────────────────────
        if metadata:
            if isinstance(metadata, dict):
                step_node = metadata.get("node")
                parents = metadata.get("parents")
            else:
                step_node = getattr(metadata, "node", None)
                parents = getattr(metadata, "parents", None)

            # Use parents dict/list as a fallback for parent_checkpoint_id
            if parents and not parent_checkpoint_id:
                if isinstance(parents, dict):
                    parent_checkpoint_id = (
                        next(iter(parents.values())) if parents else None
                    )
                elif isinstance(parents, (list, tuple)):
                    parent_checkpoint_id = parents[0] if parents else None
                else:
                    parent_checkpoint_id = str(parents)

        return checkpoint_id, parent_checkpoint_id, channel_values, step_node

    async def _write_json_log(
        self,
        thread_id: str,
        checkpoint_id: str,
        parent_checkpoint_id: str | None,
        step_node: str | None,
        channel_values: dict,
        metadata: CheckpointMetadata | None,
    ) -> None:
        """
        Write a human-readable JSON row to the `checkpoint_state_logs` audit table.

        This is best-effort — any failure is logged as a warning but does not
        raise an exception (the checkpoint write itself has already succeeded).

        Args:
            thread_id:            The conversation thread ID.
            checkpoint_id:        Unique ID for this checkpoint.
            parent_checkpoint_id: ID of the preceding checkpoint (may be None).
            step_node:            Name of the graph node that triggered this write.
            channel_values:       Full state values at this checkpoint.
            metadata:             Raw CheckpointMetadata object.
        """
        try:
            serialized_values = self._serialize_state_values(channel_values)
            serialized_metadata = self._serialize_metadata(metadata)

            async with self.pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO checkpoint_state_logs (
                        conversation_id,
                        checkpoint_id,
                        parent_checkpoint_id,
                        step_node,
                        state_values,
                        metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        thread_id,
                        checkpoint_id,
                        parent_checkpoint_id,
                        step_node,
                        json.dumps(serialized_values, default=str),
                        json.dumps(serialized_metadata, default=str),
                    ),
                )
        except Exception as e:
            logger.warning(
                "⚠️ Failed to write JSON checkpoint log: %s", e, exc_info=True
            )

    # ── LangGraph BaseCheckpointSaver interface ───────────────────────────────

    async def aget_tuple(
        self,
        config: RunnableConfig,
    ) -> Optional[CheckpointTuple]:
        """
        Fetch the latest checkpoint tuple for a given thread configuration.

        Read order: Redis (fast) → Postgres (durable fallback).
        On a Postgres hit, the result is written back to Redis to warm the cache.

        Args:
            config: LangGraph runnable config containing `configurable.thread_id`.

        Returns:
            The CheckpointTuple if found, else None.
        """
        # ── 1. Try Redis first ────────────────────────────────────────────────
        tuple_ = await self.redis_saver.aget_tuple(config)
        if tuple_ is not None:
            return tuple_

        # ── 2. Fall back to Postgres ──────────────────────────────────────────
        tuple_ = await self.postgres_saver.aget_tuple(config)
        if tuple_ is not None:
            try:
                # Warm the Redis cache so the next read is fast
                await self.redis_saver.aput(
                    tuple_.config, tuple_.checkpoint, tuple_.metadata, {}
                )
                thread_id = tuple_.config.get("configurable", {}).get("thread_id")
                await self._expire_thread_keys(thread_id)
            except Exception as e:
                logger.error("Failed to warm Redis cache: %s", e)

        return tuple_

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """
        List checkpoints for a thread, reading from Postgres (durable storage).

        Redis is not used for listing — Postgres is the source of truth for
        historical checkpoint enumeration.

        Args:
            config: Thread configuration.
            filter: Optional metadata filters.
            before: Limit results to checkpoints before this config.
            limit: Maximum number of results.

        Yields:
            Matching CheckpointTuples in reverse chronological order.
        """
        async for item in self.postgres_saver.alist(
            config, filter=filter, before=before, limit=limit
        ):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Save a checkpoint to both Postgres and Redis simultaneously.

        Write order: Postgres first (durability), then Redis (speed).
        Also writes a human-readable JSON row to `checkpoint_state_logs` if
        a connection pool is available.  Finally, sets a 1-hour TTL on all
        Redis keys for this thread to prevent unbounded memory growth.

        Args:
            config: Thread configuration.
            checkpoint: The checkpoint data to save.
            metadata: Associated checkpoint metadata.
            new_versions: Updated channel versions.

        Returns:
            The updated RunnableConfig (from the Redis saver).
        """
        # ── 1. Write to Postgres for durable persistence ──────────────────────
        await self.postgres_saver.aput(config, checkpoint, metadata, new_versions)

        # ── 2. Write to Redis for fast subsequent reads ───────────────────────
        res = await self.redis_saver.aput(config, checkpoint, metadata, new_versions)

        # ── 3. Write human-readable JSON audit log (best-effort) ──────────────
        if self.pool:
            thread_id = config.get("configurable", {}).get("thread_id")
            if thread_id:
                checkpoint_id, parent_checkpoint_id, channel_values, step_node = (
                    self._extract_checkpoint_fields(checkpoint, metadata)
                )
                await self._write_json_log(
                    thread_id=thread_id,
                    checkpoint_id=checkpoint_id,
                    parent_checkpoint_id=parent_checkpoint_id,
                    step_node=step_node,
                    channel_values=channel_values,
                    metadata=metadata,
                )

        # ── 4. Set TTL on all Redis keys for this thread ──────────────────────
        thread_id = config.get("configurable", {}).get("thread_id")
        await self._expire_thread_keys(thread_id)

        return res

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Persist intermediate write-ahead entries for a task.

        Only written to Postgres — these are intermediate entries that don't
        need to be cached in Redis since the final `aput()` will follow.

        Args:
            config: Thread configuration.
            writes: Sequence of (channel_name, value) pairs to persist.
            task_id: The LangGraph task ID for this write batch.
        """
        await self.postgres_saver.aput_writes(config, writes, task_id)

    def get_next_version(self, current: Optional[str], channel: Any) -> str:
        """
        Compute the next version string for a channel.

        Delegates to Postgres saver — version generation strategy is tied to
        the durable storage layer.

        Args:
            current: The current version string, or None for the initial version.
            channel: The channel descriptor.

        Returns:
            The next version string.
        """
        return self.postgres_saver.get_next_version(current, channel)
