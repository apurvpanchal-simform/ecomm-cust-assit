"""
Application-level response cache for the FAQ agent.

Caches the full generated answer in Redis for 5 minutes, keyed by
(customer_id, normalized_query_hash). This is separate from the global
LangChain LLM cache and works within the same chat session.
"""

import hashlib
import json
import logging
import os
import re as _re

import redis

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 300  # 5 minutes
_KEY_PREFIX = "faq_resp"

_redis_client: redis.Redis | None = None


def _get_client() -> redis.Redis:
    """Lazy-init a module-level sync Redis client."""
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
    return _redis_client


def _normalize_query(query: str) -> str:
    """Normalize a query for consistent cache key generation.

    Applies lowercase, strips punctuation, and collapses whitespace so that
    minor variations like trailing '?' or extra spaces don't cause cache misses.
    Examples:
        "What is your return policy ?" -> "what is your return policy"
        "What is your return policy?"  -> "what is your return policy"
        "what is your  return policy"  -> "what is your return policy"
    """
    lowered = query.lower()
    no_punct = _re.sub(r"[^\w\s]", "", lowered)
    collapsed = _re.sub(r"\s+", " ", no_punct).strip()
    return collapsed


def _build_key(query: str) -> str:
    """Build a global Redis key from a normalized query hash.

    FAQ responses are policy-level content identical for all users,
    so the key is NOT scoped by customer_id — one cached answer serves everyone.
    """
    normalized = _normalize_query(query)
    query_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    return f"{_KEY_PREFIX}:global:{query_hash}"



def get_faq_response(customer_id: str, query: str) -> str | None:
    """
    Look up a cached FAQ response.

    The cache is global (not scoped by customer_id) because FAQ answers are
    identical for all users. customer_id is only used for logging.

    Args:
        customer_id: Used for logging only.
        query: The raw user query string.

    Returns:
        The cached response string if a hit exists, otherwise None.
    """
    try:
        client = _get_client()
        key = _build_key(query)
        cached = client.get(key)
        if cached:
            data = json.loads(cached)
            logger.info(f"🟢 FAQ RESPONSE CACHE HIT | customer={customer_id} | key={key} | query='{query}'")
            return data.get("response")
        logger.info(f"🔴 FAQ RESPONSE CACHE MISS | customer={customer_id} | key={key} | query='{query}'")
        return None
    except Exception as e:
        logger.warning(f"FAQ cache lookup failed (non-critical): {e}")
        return None


def set_faq_response(customer_id: str, query: str, response: str) -> None:
    """
    Store an FAQ response in the global cache.

    The cache is global (not scoped by customer_id) because FAQ answers are
    identical for all users. customer_id is only used for logging.

    Args:
        customer_id: Used for logging only.
        query: The raw user query string.
        response: The LLM-generated response to cache.
    """
    try:
        client = _get_client()
        key = _build_key(query)
        client.setex(key, _CACHE_TTL_SECONDS, json.dumps({"response": response}))
        logger.info(
            f"💾 FAQ response cached globally | triggered_by={customer_id} | key={key} | query='{query}' | TTL={_CACHE_TTL_SECONDS}s"
        )
    except Exception as e:
        logger.warning(f"FAQ cache write failed (non-critical): {e}")
