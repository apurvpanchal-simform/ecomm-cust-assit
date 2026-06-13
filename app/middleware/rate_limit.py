"""
Rate limiting middleware for the chat API.
"""

from fastapi import Depends, HTTPException, Request

from app.middleware.auth import get_current_customer


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
    if not redis_client:
        return customer_id

    key = f"rate_limit:{customer_id}"

    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.incr(key)
        pipe.expire(key, 60, nx=True)
        results = await pipe.execute()

    current_count = results[0]
    if current_count > 10:
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests. Please wait a minute before trying again.",
        )

    return customer_id
