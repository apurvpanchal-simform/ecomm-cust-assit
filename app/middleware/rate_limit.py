"""
Rate limiting middleware for the chat API.
"""

from fastapi import Depends, HTTPException, Request

from app.middleware.auth import get_current_customer


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

    key = f"rate_limit:{customer_id}"

    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.incr(key)
        pipe.expire(key, 60, nx=True)
        results = await pipe.execute()

    return results[0] > limit


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
