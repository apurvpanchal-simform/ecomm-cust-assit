"""
Rate limiting middleware for the chat API.

Provides two mechanisms:
1. `check_chat_rate_limit` — per-customer **chat message** cap enforced via a
   dedicated Redis key ``rate_limit:chat:{customer_id}``.  Only chat WebSocket
   turns increment this counter, so UI background polls (conversation list,
   order data) never consume from the chat quota.

2. `rate_limit_customer` — FastAPI dependency used on REST endpoints.  Uses a
   separate ``rate_limit:rest:{customer_id}`` key with a much higher cap so
   normal UI polling is never blocked.
"""

import logging

from fastapi import Depends, HTTPException, Request

from app.config import RATE_LIMIT_CONFIG
from app.middleware.auth import get_current_customer

logger = logging.getLogger(__name__)

# REST endpoints get a much higher cap — they are background polls, not chat.
_REST_LIMIT_MULTIPLIER = 10


# ── Shared ban-check helper ───────────────────────────────────────────────────


async def _check_ban(redis_client, customer_id: str) -> None:
    """Raise 403 immediately if the customer is currently banned."""
    ban_key = f"banned:{customer_id}"
    if await redis_client.get(ban_key):
        raise HTTPException(
            status_code=403,
            detail="You have been temporarily banned for spamming.",
        )


# ── Core counter helper ───────────────────────────────────────────────────────


async def _increment_and_check(
    redis_client,
    counter_key: str,
    customer_id: str,
    limit: int,
    scope: str,
) -> bool:
    """
    Increment the Redis counter for *counter_key* and enforce limits.

    Args:
        redis_client: Async Redis client.
        counter_key:  The Redis key to increment (scoped per customer+type).
        customer_id:  Used for ban-key construction.
        limit:        Soft cap — returns True (rate-limited) when count >= limit.
        scope:        Human-readable label used in log messages ('chat'/'rest').

    Returns:
        True  — limit exceeded, caller should reject the request.
        False — request is within the allowed limit.

    Raises:
        HTTPException 403: Customer is already banned OR just crossed spam threshold.
    """
    # ── 1. Reject immediately if already banned ───────────────────────────────
    await _check_ban(redis_client, customer_id)

    # ── 2. Increment the scoped per-minute counter (atomic pipeline) ──────────
    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.incr(counter_key)
        # nx=True: EXPIRE is only set on the very first write so the window
        # starts fresh after 60 seconds, not on every request.
        pipe.expire(counter_key, 60, nx=True)
        results = await pipe.execute()

    count = results[0]
    logger.debug(
        "[rate_limit] scope=%s | customer=%s | count=%d | limit=%d | spam_threshold=%d",
        scope, customer_id, count, limit, RATE_LIMIT_CONFIG.spam_threshold,
    )

    # ── 3. Hard ban when spam threshold is crossed ────────────────────────────
    if count >= RATE_LIMIT_CONFIG.spam_threshold:
        ban_key = f"banned:{customer_id}"
        await redis_client.setex(ban_key, RATE_LIMIT_CONFIG.ban_duration_seconds, "1")
        logger.warning(
            "[rate_limit] BANNED customer=%s after %d requests in 60s (scope=%s)",
            customer_id, count, scope,
        )
        raise HTTPException(
            status_code=403,
            detail="You have been temporarily banned for spamming.",
        )

    # ── 4. Soft rate-limit signal (>= so limit=10 blocks on the 10th request) ─
    if count >= limit:
        logger.info(
            "[rate_limit] LIMIT EXCEEDED scope=%s | customer=%s | count=%d >= limit=%d",
            scope, customer_id, count, limit,
        )
        return True

    return False


# ── Public: chat WebSocket rate limiter ──────────────────────────────────────


async def check_chat_rate_limit(redis_client, customer_id: str) -> bool:
    """
    Check whether a customer has exceeded their per-minute **chat** allowance.

    Uses the key ``rate_limit:chat:{customer_id}`` — completely separate from
    REST endpoint counters so background UI polls never eat into the chat quota.

    Args:
        redis_client: Async Redis client instance (may be None in local dev).
        customer_id:  The authenticated customer's ID.

    Returns:
        True  — chat limit exceeded (caller should send a 429 error frame).
        False — request is within the allowed chat limit.

    Raises:
        HTTPException 403: If the customer is currently banned.
    """
    if not redis_client:
        return False  # No Redis — skip rate limiting in local dev

    key = f"rate_limit:chat:{customer_id}"
    return await _increment_and_check(
        redis_client, key, customer_id,
        limit=RATE_LIMIT_CONFIG.normal_limit,
        scope="chat",
    )


# ── Public: REST endpoint dependency ─────────────────────────────────────────


async def rate_limit_customer(
    request: Request,
    customer_id: str = Depends(get_current_customer),
) -> str:
    """
    FastAPI dependency that enforces per-customer rate limiting on REST routes.

    Uses a separate ``rate_limit:rest:{customer_id}`` key with a much higher
    cap (``normal_limit × 10``) so normal UI background polling is never
    inadvertently blocked.

    Args:
        request:     The incoming FastAPI request (used to access app state).
        customer_id: Extracted from the JWT by `get_current_customer`.

    Returns:
        The customer ID if the request is within the rate limit.

    Raises:
        HTTPException 429: Too many requests — ask the caller to back off.
        HTTPException 403: Customer is banned.
    """
    redis_client = getattr(request.app.state, "redis", None)
    if not redis_client:
        return customer_id  # No Redis — skip rate limiting in local dev

    key = f"rate_limit:rest:{customer_id}"
    rest_limit = RATE_LIMIT_CONFIG.normal_limit * _REST_LIMIT_MULTIPLIER

    if await _increment_and_check(redis_client, key, customer_id, limit=rest_limit, scope="rest"):
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests. Please wait a minute before trying again.",
        )

    return customer_id

