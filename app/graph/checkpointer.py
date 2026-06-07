from typing import Optional, Any, AsyncIterator, Dict, Sequence, Tuple
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver, 
    Checkpoint, 
    CheckpointMetadata, 
    CheckpointTuple, 
    ChannelVersions
)

import logging

logger = logging.getLogger(__name__)

class AsyncDualCheckpointer(BaseCheckpointSaver):
    """
    An async LangGraph checkpointer that writes to both Redis and Postgres (Supabase).
    Reads prioritize Redis for speed, falling back to Postgres if a cache miss occurs.
    """
    def __init__(self, redis_saver: BaseCheckpointSaver, postgres_saver: BaseCheckpointSaver):
        super().__init__()
        self.redis_saver = redis_saver
        self.postgres_saver = postgres_saver

    async def aget_tuple(
        self,
        config: RunnableConfig,
    ) -> Optional[CheckpointTuple]:

        thread_id = (
            config.get("configurable", {})
            .get("thread_id")
        )

        logger.debug(f"THREAD_ID={thread_id}")

        tuple_ = await self.redis_saver.aget_tuple(config)

        if tuple_ is not None:
            logger.debug("⚡ REDIS HIT")
            return tuple_

        logger.debug("❌ REDIS MISS")

        tuple_ = await self.postgres_saver.aget_tuple(config)

        if tuple_ is not None:
            logger.debug("🐘 POSTGRES HIT")
        else:
            logger.debug("❌ POSTGRES MISS")

        return tuple_

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        # Read history from Postgres since it's our durable long-term storage
        async for item in self.postgres_saver.alist(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        # Write the checkpoint to Supabase for long-term persistence.
        await self.postgres_saver.aput(config, checkpoint, metadata, new_versions)
        # Write the latest checkpoint to Redis (hot cache).
        res = await self.redis_saver.aput(config, checkpoint, metadata, new_versions)
        
        # Prune old checkpoints from Redis — keep only the latest one.
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if thread_id and hasattr(self.redis_saver, "_redis"):
                redis_client = self.redis_saver._redis
                latest_id = checkpoint["id"]
                
                async for key in redis_client.scan_iter(match=f"*{thread_id}*"):
                    key_str = key.decode()
                    # Keep checkpoint_latest pointer and keys belonging to the current checkpoint
                    if "checkpoint_latest" in key_str or latest_id in key_str:
                        await redis_client.expire(key, 3600)  # 1-hour TTL
                        continue
                    await redis_client.delete(key)
        except Exception as e:
            logger.error(f"Failed to prune Redis keys for thread {thread_id}: {e}")
            
        return res

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        # Only write intermediate states to Postgres.
        await self.postgres_saver.aput_writes(config, writes, task_id)

    def get_next_version(self, current: Optional[str], channel: Any) -> str:
        # Delegate version generation to one of the underlying savers
        return self.postgres_saver.get_next_version(current, channel)