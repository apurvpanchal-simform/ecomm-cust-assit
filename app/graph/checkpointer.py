from typing import Optional, Any, Iterator, Dict, Sequence, Tuple
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver, 
    Checkpoint, 
    CheckpointMetadata, 
    CheckpointTuple, 
    ChannelVersions
)

class DualCheckpointer(BaseCheckpointSaver):
    """
    A custom LangGraph checkpointer that writes to both Redis and Postgres (Supabase).
    Reads prioritize Redis for speed, falling back to Postgres if a cache miss occurs.
    """
    def __init__(self, redis_saver: BaseCheckpointSaver, postgres_saver: BaseCheckpointSaver):
        super().__init__()
        self.redis_saver = redis_saver
        self.postgres_saver = postgres_saver

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        # 1. Try fetching from Redis first. 
        # This is where the magic happens for speed: reading from RAM is microseconds fast.
        tuple_ = self.redis_saver.get_tuple(config)
        if tuple_ is not None:
            return tuple_
            
        # 2. Fall back to Postgres if not found in Redis (e.g., if Redis data expired or was evicted).
        # This involves a disk read, which is slower, but guarantees we don't lose old chats.
        return self.postgres_saver.get_tuple(config)

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        # Read history from Postgres since it's our durable long-term storage
        # Redis might not contain the full history of every single conversation.
        return self.postgres_saver.list(config, filter=filter, before=before, limit=limit)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        # Write the checkpoint to Supabase for long-term persistence.
        self.postgres_saver.put(config, checkpoint, metadata, new_versions)
        # Write the checkpoint to Redis and return the resulting config.
        return self.redis_saver.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        # Write intermediate states/writes to both storage backends.
        self.postgres_saver.put_writes(config, writes, task_id)
        self.redis_saver.put_writes(config, writes, task_id)

    def get_next_version(self, current: Optional[str], channel: Any) -> str:
        # Delegate version generation to one of the underlying savers
        return self.postgres_saver.get_next_version(current, channel)