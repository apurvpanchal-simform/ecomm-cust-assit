import asyncio
import time


class RedisTokenBucket:
    """
    A distributed Token Bucket rate limiter using Redis Lua scripts.
    Allows smoothing out bursts while respecting an overall Rate Limit (RPM/RPS)
    across multiple workers.
    """
    def __init__(self, redis_client, name: str, capacity: int, fill_rate: float):
        """
        Args:
            redis_client: Async redis client
            name: Name of the bucket
            capacity: Maximum burst size (e.g., 30 tokens)
            fill_rate: Tokens added per second (e.g., 30 tokens / 60 seconds = 0.5)
        """
        self.redis = redis_client
        self.name = f"token_bucket:{name}"
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens_key = f"{self.name}:tokens"
        self.ts_key = f"{self.name}:last_ts"

    async def acquire(self) -> float:
        """
        Attempts to acquire a single token.
        Returns:
            The amount of time (in seconds) the caller should sleep/wait before proceeding.
            Returns 0.0 if the token was acquired immediately.
        """
        if not self.redis:
            return 0.0
            
        lua_script = """
        local tokens_key = KEYS[1]
        local ts_key = KEYS[2]
        local capacity = tonumber(ARGV[1])
        local fill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local requested = 1

        local last_tokens = tonumber(redis.call("GET", tokens_key))
        if last_tokens == nil then
            last_tokens = capacity
        end

        local last_ts = tonumber(redis.call("GET", ts_key))
        if last_ts == nil then
            last_ts = now
        end

        local delta = math.max(0, now - last_ts)
        local current_tokens = math.min(capacity, last_tokens + delta * fill_rate)
        
        local wait_time = 0
        if current_tokens < requested then
            local tokens_needed = requested - current_tokens
            wait_time = tokens_needed / fill_rate
            current_tokens = 0
            redis.call("SET", tokens_key, current_tokens)
            redis.call("SET", ts_key, now + wait_time)
        else
            current_tokens = current_tokens - requested
            redis.call("SET", tokens_key, current_tokens)
            redis.call("SET", ts_key, now)
        end
        
        -- Expire keys to prevent memory leaks if unused
        local expire_time = math.ceil(capacity / fill_rate) * 2
        redis.call("EXPIRE", tokens_key, expire_time)
        redis.call("EXPIRE", ts_key, expire_time)
        
        return tostring(wait_time)
        """
        wait_time_str = await self.redis.eval(
            lua_script, 2, self.tokens_key, self.ts_key, self.capacity, self.fill_rate, time.time()
        )
        return float(wait_time_str)

