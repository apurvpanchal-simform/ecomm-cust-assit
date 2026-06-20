"""
Rate limiting middleware for the chat API.
"""

import asyncio
import logging

from fastapi import Depends, HTTPException, Request
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.callbacks.manager import adispatch_custom_event

from app.middleware.auth import get_current_customer

logger = logging.getLogger(__name__)


async def check_rate_limit(redis_client, customer_id: str, limit: int = 10) -> bool:
    """
    Checks if a customer has exceeded their rate limit.

    Args:
        redis_client: The Redis client instance.
        customer_id: The customer ID to check.
        limit: The maximum number of requests allowed per minute.

    Returns:
        bool: True if the limit is exceeded, False otherwise.
    """
    if not redis_client:
        return False

    ban_key = f"banned:{customer_id}"
    is_banned = await redis_client.get(ban_key)
    if is_banned:
        raise HTTPException(status_code=403, detail="You have been temporarily banned for spamming.")

    key = f"rate_limit:{customer_id}"

    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.incr(key)
        pipe.expire(key, 60, nx=True)
        results = await pipe.execute()

    requests_in_minute = results[0]

    # Spam threshold: if they hit 20 requests in a minute, ban them for 24 hours
    if requests_in_minute >= 20:
        await redis_client.setex(ban_key, 86400, "1")
        raise HTTPException(status_code=403, detail="You have been temporarily banned for spamming.")

    return requests_in_minute > limit


async def rate_limit_customer(
    request: Request,
    customer_id: str = Depends(get_current_customer),
) -> str:
    """
    FastAPI dependency to enforce rate limiting on chat requests.

    Limits each customer to a specific number of requests per minute using Redis.

    Args:
        request: The FastAPI request object.
        customer_id: The authenticated customer ID.

    Returns:
        The customer ID if the rate limit has not been exceeded.

    Raises:
        HTTPException: If the customer has exceeded the rate limit.
    """
    redis_client = getattr(request.app.state, "redis", None)

    if await check_rate_limit(redis_client, customer_id):
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests. Please wait a minute before trying again.",
        )

    return customer_id


class RateLimitCallbackHandler(AsyncCallbackHandler):
    """
    A LangChain callback handler that intercepts LLM calls and acquires 
    a slot from the RedisTokenBucket rate limiter. If it has to wait, 
    it streams a status message back to the user via custom events.
    """
    def __init__(self, token_bucket):
        self.token_bucket = token_bucket

    async def on_chat_model_start(self, serialized, messages, **kwargs):
        if not self.token_bucket:
            return
            
        wait_time = await self.token_bucket.acquire()
        
        # Get the name of the LLM/Agent if available
        agent_name = kwargs.get("name", "Unknown Agent")
        
        if wait_time > 0:
            logger.info("LLM Queue: Burst capacity reached. Waiting %.1fs for %s.", wait_time, agent_name)
            # We add a buffer of 0.5s to be safe
            wait_time += 0.5
            msg = f"High traffic... Waiting {wait_time:.1f}s for LLM capacity..."
            
            # Dispatch event to the UI so the user knows they are waiting in the queue
            await adispatch_custom_event("queue_wait_start", {"message": msg})
            await asyncio.sleep(wait_time)
            await adispatch_custom_event("queue_wait_end", {"message": "Resumed!"})
            logger.info("LLM Queue: Resumed and token acquired for %s after %.1fs wait.", agent_name, wait_time)
        else:
            logger.info("LLM Queue: Token acquired immediately for %s.", agent_name)

