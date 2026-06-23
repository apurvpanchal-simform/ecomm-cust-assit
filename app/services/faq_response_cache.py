"""
Application-level semantic response cache for the FAQ agent.

Stores fully-generated FAQ answers in Redis and retrieves them using
Vector Similarity Search (VSS) so that semantically equivalent questions
(not just exact-match queries) can be served from cache.

Cache key strategy
------------------
Each entry is a Redis Hash stored under `faq_resp:sem:<uuid>` and indexed
by a RediSearch VSS index (`idx:faq_cache`).  A KNN=1 query is run at read
time; a hit is declared when the Cosine *distance* is <= (1 - threshold).

TTL: controlled by `_CACHE_TTL_SECONDS` (default 1 hour).
Threshold: set via SEARCH_CONFIG.faq_cache_threshold (default 0.75 similarity).
"""

import logging
import os
import re as _re
import struct
import time
import uuid

import redis
from redis.commands.search.query import Query

from app.config import SEARCH_CONFIG
from app.services.llm_factory import get_embeddings

logger = logging.getLogger(__name__)

# How long a cached entry lives in Redis before being auto-expired
_CACHE_TTL_SECONDS = 3600  # 1 hour

# Module-level singletons — lazily initialised on first use
_redis_vector_client: redis.Redis | None = None
_index_ensured = False
_embeddings_512 = None


# ── Private helpers ───────────────────────────────────────────────────────────


def _get_embeddings_model():
    """
    Lazily instantiate and return the 512-dimension embedding model.

    Uses a module-level singleton so the model is only created once per process.
    """
    global _embeddings_512
    if _embeddings_512 is None:
        _embeddings_512 = get_embeddings(512)
    return _embeddings_512


def _normalize_query(query: str) -> str:
    """
    Normalise a query string for consistent embedding generation.

    Strips case, removes punctuation, and collapses whitespace so that
    "What is your return policy?" and "what is your return policy" produce
    the same embedding and therefore cache-match each other.

    Args:
        query: Raw user query string.

    Returns:
        Cleaned, lowercase, whitespace-collapsed query string.
    """
    lowered = query.lower()
    no_punct = _re.sub(r"[^\w\s]", "", lowered)
    collapsed = _re.sub(r"\s+", " ", no_punct).strip()
    return collapsed


def _get_vector_client() -> redis.Redis:
    """
    Lazily initialise and return a sync Redis client configured for binary vectors.

    `decode_responses=False` is required so that binary vector buffers are
    stored and retrieved correctly (they cannot be decoded as UTF-8 strings).
    """
    global _redis_vector_client
    if _redis_vector_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _redis_vector_client = redis.Redis.from_url(redis_url, decode_responses=False)
    return _redis_vector_client


def _ensure_index() -> None:
    """
    Lazily create the RediSearch VSS index `idx:faq_cache` if it does not exist.

    Runs at most once per process (guarded by `_index_ensured`). The index is
    created over all Hash keys with the prefix `faq_resp:sem:` and supports
    KNN searches using COSINE distance on a 512-dimensional FLOAT32 vector.
    """
    global _index_ensured
    if _index_ensured:
        return  # Already verified during this process lifetime

    client = _get_vector_client()
    index_name = "idx:faq_cache"

    try:
        # If `.info()` succeeds, the index already exists — nothing to do
        client.ft(index_name).info()
        _index_ensured = True
    except Exception:
        # Index does not exist — create it now
        from redis.commands.search.field import NumericField, TextField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        schema = [
            TextField("query"),
            TextField("response"),
            NumericField("created_at"),
            VectorField(
                "vector",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": 512,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        ]
        try:
            client.ft(index_name).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=["faq_resp:sem:"], index_type=IndexType.HASH
                ),
            )
            _index_ensured = True
            logger.info("Created RediSearch VSS index 'idx:faq_cache' successfully.")
        except Exception as e:
            logger.warning(
                "Could not create RediSearch VSS index (Search module might be disabled): %s",
                e,
            )


def _decode_doc_field(value) -> str:
    """
    Decode a Redis document field that may be returned as bytes or str.

    RediSearch returns field values as bytes when `decode_responses=False`,
    so this helper ensures we always work with plain Python strings.

    Args:
        value: The raw field value from a RediSearch document.

    Returns:
        A decoded UTF-8 string.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _build_knn_query(vector_bytes: bytes) -> Query:
    """
    Build a RediSearch KNN=1 query object for the given binary vector.

    Searches for the single nearest neighbour by COSINE distance and returns
    the `query`, `response`, and computed `score` fields.

    Args:
        vector_bytes: IEEE 754 FLOAT32 packed bytes of the query vector.

    Returns:
        A configured RediSearch Query object ready to be executed.
    """
    return (
        Query("*=>[KNN 1 @vector $vector_val AS score]")
        .sort_by("score")
        .return_fields("query", "response", "score")
        .dialect(2)
    )


# ── Public cache API ──────────────────────────────────────────────────────────


async def get_faq_response(customer_id: str, query: str) -> str | None:
    """
    Look up a cached FAQ response using semantic vector similarity.

    Embeds the query, runs a KNN=1 search against the Redis VSS index, and
    returns the cached answer if the nearest neighbour's cosine similarity
    meets the configured threshold.

    Similarity threshold is read from SEARCH_CONFIG.faq_cache_threshold.
    Cosine *distance* returned by Redis = 1 − similarity; hit when distance ≤ 1 - threshold.

    Args:
        customer_id: Used for logging only (not part of the cache key).
        query: The raw user query string.

    Returns:
        The cached response string on a semantic hit, or None on a miss/error.
    """
    try:
        client = _get_vector_client()
        _ensure_index()

        # ── 1. Embed and serialise the normalised query ───────────────────────
        embeddings = _get_embeddings_model()
        normalized_query = _normalize_query(query)
        vector = await embeddings.aembed_query(normalized_query)
        # Pack float list into IEEE 754 FLOAT32 binary format required by RedisSearch
        vector_bytes = struct.pack(f"{len(vector)}f", *vector)

        # ── 2. Run KNN=1 similarity search ────────────────────────────────────
        index_name = "idx:faq_cache"
        q = _build_knn_query(vector_bytes)
        results = client.ft(index_name).search(
            q, query_params={"vector_val": vector_bytes}
        )

        if not results.docs:
            logger.info(
                "🔴 FAQ SEMANTIC CACHE MISS (No index hits) | customer=%s | query='%s'",
                customer_id,
                query,
            )
            return None

        # ── 3. Evaluate the nearest-neighbour's score against the threshold ───
        doc = results.docs[0]
        score = float(doc.score)  # Cosine distance: 0.0 = identical, 1.0 = opposite

        threshold_similarity = SEARCH_CONFIG.faq_cache_threshold
        max_distance = 1.0 - threshold_similarity  # e.g. 0.75 sim → 0.25 max dist

        closest_query = _decode_doc_field(doc.query)

        if score <= max_distance:
            # ── Cache HIT ─────────────────────────────────────────────────────
            response_val = _decode_doc_field(doc.response)
            logger.info(
                "🟢 FAQ SEMANTIC CACHE HIT | customer=%s | score=%.4f (sim=%.4f, threshold=%.2f) | matched_query='%s' | query='%s'",
                customer_id,
                score,
                1 - score,
                threshold_similarity,
                closest_query,
                query,
            )
            return response_val
        else:
            # ── Cache MISS — closest neighbour is not similar enough ──────────
            logger.info(
                "🔴 FAQ SEMANTIC CACHE MISS (Below threshold: %.2f) | customer=%s | closest_score=%.4f (sim=%.4f) | closest_query='%s' | query='%s'",
                threshold_similarity,
                customer_id,
                score,
                1 - score,
                closest_query,
                query,
            )
            return None

    except redis.exceptions.ResponseError as e:
        if "No such index" in str(e):
            logger.warning(
                "RediSearch Index missing: %s. Semantic cache is disabled.", e
            )
        else:
            logger.warning("Redis ResponseError in FAQ cache lookup: %s", e)
        return None
    except Exception as e:
        logger.warning("FAQ cache lookup failed (non-critical): %s", e, exc_info=True)
        return None


async def set_faq_response(customer_id: str, query: str, response: str) -> None:
    """
    Store an FAQ response in the Redis VSS cache.

    Embeds the query, serialises the vector to FLOAT32 bytes, and writes a
    Redis Hash entry under a unique `faq_resp:sem:<uuid>` key. The key is
    set to expire after `_CACHE_TTL_SECONDS` (default: 1 hour).

    Args:
        customer_id: Used for logging only.
        query: The raw user query string.
        response: The fully-generated LLM response to cache.
    """
    try:
        client = _get_vector_client()
        _ensure_index()

        # ── 1. Embed and serialise the query ──────────────────────────────────
        embeddings = _get_embeddings_model()
        normalized_query = _normalize_query(query)
        vector = await embeddings.aembed_query(normalized_query)
        vector_bytes = struct.pack(f"{len(vector)}f", *vector)

        # ── 2. Write the Hash entry to Redis ──────────────────────────────────
        key = f"faq_resp:sem:{uuid.uuid4().hex}"
        client.hset(
            key,
            mapping={
                "query": query,
                "response": response,
                "created_at": str(time.time()),
                "vector": vector_bytes,
            },
        )

        # ── 3. Set TTL so stale entries are auto-removed ───────────────────────
        client.expire(key, _CACHE_TTL_SECONDS)
        logger.info(
            "💾 FAQ response cached semantically | triggered_by=%s | key=%s | query='%s' | TTL=%ss",
            customer_id,
            key,
            query,
            _CACHE_TTL_SECONDS,
        )

    except redis.exceptions.ResponseError as e:
        if "No such index" in str(e):
            logger.warning(
                "RediSearch Index missing: %s. Cannot write to semantic cache.", e
            )
        else:
            logger.warning("Redis ResponseError in FAQ cache write: %s", e)
    except Exception as e:
        logger.warning("FAQ cache write failed (non-critical): %s", e, exc_info=True)
