"""
Custom LangGraph Checkpointer that implements a Dual-Write strategy.

Writes to both an ephemeral Redis cache for fast hot-path retrieval,
and a durable Postgres (Supabase) database for long-term persistence and auditing.
"""

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
    An async LangGraph checkpointer that writes to both Redis and Postgres (Supabase).
    Reads prioritize Redis for speed, falling back to Postgres if a cache miss occurs.
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

    async def _expire_thread_keys(self, thread_id: str, ttl: int = 3600):
        """Scans and expires all LangGraph checkpoint keys associated with a thread."""
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
                    async with redis_client.pipeline(transaction=False) as pipe:
                        for key in keys:
                            pipe.expire(key, ttl)
                        await pipe.execute()
                if cursor == 0:
                    break
        except Exception as e:
            logger.error("Failed to set Redis TTL for thread %s: %s", thread_id, e)

    async def aget_tuple(
        self,
        config: RunnableConfig,
    ) -> Optional[CheckpointTuple]:
        """
        Fetch the checkpoint tuple for a given configuration.

        Attempts to read from Redis first. On cache miss, falls back to Postgres,
        and if found, re-warms the Redis cache.

        Args:
            config: The thread configuration containing thread_id.

        Returns:
            The CheckpointTuple if found, else None.
        """

        thread_id = config.get("configurable", {}).get("thread_id")

        logger.info("THREAD_ID=%s", thread_id)

        tuple_ = await self.redis_saver.aget_tuple(config)

        if tuple_ is not None:
            logger.info("⚡ REDIS READ HIT (aget_tuple)")
            return tuple_

        logger.info("❌ REDIS READ MISS (aget_tuple)")

        tuple_ = await self.postgres_saver.aget_tuple(config)

        if tuple_ is not None:
            logger.info("🐘 SUPABASE READ HIT (aget_tuple)")
            try:
                # WARM REDIS CACHE: Write the tuple back to Redis so subsequent reads hit the cache
                logger.info("⚡ REDIS WRITE (Warm Cache)")
                await self.redis_saver.aput(
                    tuple_.config, tuple_.checkpoint, tuple_.metadata, {}
                )

                # Set TTL on all restored keys
                thread_id = tuple_.config.get("configurable", {}).get("thread_id")
                await self._expire_thread_keys(thread_id)
            except Exception as e:
                logger.error("Failed to warm Redis cache: %s", e)
        else:
            logger.info("❌ SUPABASE READ MISS (aget_tuple)")

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
        List checkpoints for a given thread configuration.

        Reads from Postgres as it serves as the long-term durable storage layer.

        Args:
            config: The thread configuration.
            filter: Optional filters.
            before: Limit to checkpoints before this one.
            limit: Maximum number of checkpoints to return.

        Yields:
            Matching CheckpointTuples.
        """
        # Read history from Postgres since it's our durable long-term storage
        logger.info("🐘 SUPABASE READ (alist)")
        async for item in self.postgres_saver.alist(
            config, filter=filter, before=before, limit=limit
        ):
            yield item

    def _serialize_message(self, msg: Any) -> Any:
        if not msg:
            return msg
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
        if isinstance(msg, dict):
            return {k: self._serialize_message(v) for k, v in msg.items()}
        if isinstance(msg, (list, tuple)):
            return [self._serialize_message(x) for x in msg]
        return msg

    def _serialize_state_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values:
            return {}
        serialized = {}
        for k, v in values.items():
            if k in ("image_base64", "image_embedding"):
                serialized[k] = "[EXCLUDED_FOR_SIZE]"
                continue

            # Use deep serialization for everything
            serialized[k] = self._serialize_message(v)
        return serialized

    def _serialize_metadata(self, metadata: Any) -> Dict[str, Any]:
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
            import json

            json.dumps(metadata)
            return metadata
        except TypeError:
            return {"raw": str(metadata)}

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """
        Save a checkpoint to both underlying storage systems.

        Writes simultaneously to Postgres and Redis. Also writes a human-readable
        JSON log to a custom `checkpoint_state_logs` table for debugging purposes.
        Expires old Redis checkpoints to save memory.

        Args:
            config: The thread configuration.
            checkpoint: The checkpoint data.
            metadata: Associated metadata.
            new_versions: Channel versions.

        Returns:
            The updated RunnableConfig.
        """
        # Write the checkpoint to Supabase for long-term persistence.
        logger.info("🐘 SUPABASE WRITE (aput)")
        await self.postgres_saver.aput(config, checkpoint, metadata, new_versions)
        # Write the latest checkpoint to Redis (hot cache).
        logger.info("⚡ REDIS WRITE (aput)")
        res = await self.redis_saver.aput(config, checkpoint, metadata, new_versions)

        # Write readable JSON log entry if connection pool is available
        if self.pool:
            try:
                thread_id = config.get("configurable", {}).get("thread_id")
                if thread_id:
                    # Extract checkpoint details
                    checkpoint_id = ""
                    parent_checkpoint_id = None
                    channel_values = {}

                    if checkpoint:
                        if isinstance(checkpoint, dict):
                            channel_values = checkpoint.get("channel_values", {})
                            checkpoint_id = checkpoint.get("id", "")
                            parent_checkpoint_id = checkpoint.get(
                                "parent_checkpoint_id"
                            )
                        else:
                            channel_values = getattr(checkpoint, "channel_values", {})
                            checkpoint_id = getattr(checkpoint, "id", "")
                            parent_checkpoint_id = getattr(
                                checkpoint, "parent_checkpoint_id", None
                            )

                    step_node = None
                    if metadata:
                        if isinstance(metadata, dict):
                            step_node = metadata.get("node")
                            parents = metadata.get("parents")
                        else:
                            step_node = getattr(metadata, "node", None)
                            parents = getattr(metadata, "parents", None)

                        if parents and not parent_checkpoint_id:
                            if isinstance(parents, dict):
                                parent_checkpoint_id = (
                                    next(iter(parents.values())) if parents else None
                                )
                            elif isinstance(parents, (list, tuple)):
                                parent_checkpoint_id = parents[0] if parents else None
                            else:
                                parent_checkpoint_id = str(parents)

                    # Serialize values
                    serialized_values = self._serialize_state_values(channel_values)
                    serialized_metadata = self._serialize_metadata(metadata)

                    import json

                    logger.info(
                        logger.info(
                            "💾 Writing human-readable JSON state log for thread_id=%s, node=%s",
                            thread_id,
                            step_node,
                        )
                    )

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
                    logger.error(
                        "⚠️ Failed to write JSON checkpoint log: %s", e, exc_info=True
                    )
                )

        # Prune old checkpoints from Redis — keep only the latest one.
        # We apply a 1-hour TTL to ALL LangGraph keys associated with this thread.
        thread_id = config.get("configurable", {}).get("thread_id")
        await self._expire_thread_keys(thread_id)

        return res

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Asynchronously writes data according to the given RunnableConfig.

        Args:
            config: Configuration for the runnable operation.

        Returns:
            None. The operation completes when all writes are finished.
        """
        # Only write intermediate states to Postgres.
        logger.info("🐘 SUPABASE WRITE (aput_writes)")
        await self.postgres_saver.aput_writes(config, writes, task_id)

    def get_next_version(self, current: Optional[str], channel: Any) -> str:
        """Return the next version string derived from the current version and channel. If current is None, compute the initial version for the channel."""
        # Delegate version generation to one of the underlying savers
        return self.postgres_saver.get_next_version(current, channel)
