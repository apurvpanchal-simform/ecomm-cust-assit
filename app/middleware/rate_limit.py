"""
Rate limiting middleware for the chat API.

Provides one mechanism:
1. `check_rate_limit` / `rate_limit_customer` — per-customer request-per-minute
   cap enforced via a Redis counter.  Customers exceeding the spam threshold
   are temporarily banned for 24 hours.
"""

import logging

from fastapi import Depends, HTTPException, Request

from app.middleware.auth import get_current_customer

logger = logging.getLogger(__name__)


# ── Per-minute request-count rate limiter ─────────────────────────────────────


async def check_rate_limit(redis_client, customer_id: str, limit: int = 10) -> bool:
    """
    Check whether a customer has exceeded their per-minute request allowance.

    Uses a simple Redis INCR + EXPIRE counter.  If the customer has already
    been banned, a 403 is raised immediately.  If this request pushes their
    count to ≥ 20 in a single minute (spam threshold), they are banned for
    24 hours and a 403 is raised.

    Args:
        redis_client: Async Redis client instance (may be None in tests).
        customer_id: The authenticated customer's ID.
        limit: Normal per-minute cap before a 429 is returned (default 10).

    Returns:
        True if the rate limit is exceeded (caller should return 429).
        False if the request is within the limit.

    Raises:
        HTTPException 403: If the customer is currently banned.
        HTTPException 403: If this request crosses the spam threshold (≥ 20/min).
    """
    if not redis_client:
        # No Redis configured — skip rate limiting (e.g. local dev)
        return False

    # ── 1. Check if this customer is already banned ───────────────────────────
    ban_key = f"banned:{customer_id}"
    is_banned = await redis_client.get(ban_key)
    if is_banned:
        raise HTTPException(
            status_code=403,
            detail="You have been temporarily banned for spamming.",
        )

    # ── 2. Increment the per-minute counter (atomic pipeline) ────────────────
    key = f"rate_limit:{customer_id}"
    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.incr(key)
        # nx=True means EXPIRE is only set on first write (i.e. resets every minute)
        pipe.expire(key, 60, nx=True)
        results = await pipe.execute()

    requests_in_minute = results[0]

    # ── 3. Enforce spam ban (hard cap at 20 req/min) ─────────────────────────
    if requests_in_minute >= 20:
        await redis_client.setex(ban_key, 86400, "1")  # 86400s = 24 hours
        raise HTTPException(
            status_code=403,
            detail="You have been temporarily banned for spamming.",
        )

    # ── 4. Return soft rate-limit signal (caller returns 429) ────────────────
    return requests_in_minute > limit


async def rate_limit_customer(
    request: Request,
    customer_id: str = Depends(get_current_customer),
) -> str:
    """
    FastAPI dependency that enforces per-customer rate limiting on chat routes.

    Reads the Redis client from `request.app.state.redis` and delegates to
    `check_rate_limit`.  If the limit is exceeded, raises a 429 response.

    Args:
        request: The incoming FastAPI request (used to access app state).
        customer_id: Extracted from the JWT by `get_current_customer`.

    Returns:
        The customer ID if the request is within the rate limit.

    Raises:
        HTTPException 429: Too many requests — ask the user to wait.
        HTTPException 403: Customer is banned (propagated from `check_rate_limit`).
    """
    redis_client = getattr(request.app.state, "redis", None)

    if await check_rate_limit(redis_client, customer_id):
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests. Please wait a minute before trying again.",
        )

    return customer_id

