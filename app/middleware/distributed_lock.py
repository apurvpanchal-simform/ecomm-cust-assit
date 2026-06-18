import asyncio
import time
import uuid


class RedisSemaphore:
    """
    A robust distributed semaphore using Redis Sorted Sets (ZSET).
    
    This implementation safely throttles concurrent executions across multiple distributed workers.
    It automatically handles crashed workers by enforcing a strict TTL timeout on held tokens.
    """

    def __init__(self, redis_client, name: str, limit: int, timeout: int = 60):
        self.redis = redis_client
        self.name = f"semaphore:{name}"
        self.limit = limit
        self.timeout = timeout
        self.identifier = str(uuid.uuid4())

    async def acquire(self) -> bool:
        """Wait until a slot is available in the semaphore."""
        while True:
            now = time.time()
            
            # 1. Clean up any expired tokens from crashed workers
            await self.redis.zremrangebyscore(self.name, "-inf", now - self.timeout)
            
            # 2. Add our identifier to the set with the current timestamp
            await self.redis.zadd(self.name, {self.identifier: now})
            
            # 3. Check our rank (position in the queue)
            rank = await self.redis.zrank(self.name, self.identifier)
            
            if rank is not None and rank < self.limit:
                return True
                
            # Otherwise, no slot is available. Remove ourselves from the set so we don't clog it.
            await self.redis.zrem(self.name, self.identifier)
            
            # Wait for half a second before trying again (Polling)
            await asyncio.sleep(0.5)

    async def release(self):
        """Release the slot back into the semaphore."""
        await self.redis.zrem(self.name, self.identifier)

    async def __aenter__(self):
        """Enable 'async with RedisSemaphore(...)' syntax."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure the token is always released even if an exception occurs."""
        await self.release()

class DummySemaphore:
    """A no-op semaphore for when Redis is disabled."""
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
