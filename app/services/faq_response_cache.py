"""
Application-level semantic response cache for the FAQ agent.

Caches the full generated answer in Redis for 5 minutes, using Redis Vector Similarity Search (VSS)
to match queries semantically (Cosine Similarity >= 0.90, i.e. Cosine Distance <= 0.10).
"""

import logging
import os
import struct
import time
import uuid

import redis
from redis.commands.search.query import Query

from app.services.llm_factory import get_embeddings

logger = logging.getLogger(__name__)

import re as _re

_CACHE_TTL_SECONDS = 3600  # 1 hour
_redis_vector_client: redis.Redis | None = None
_index_ensured = False
_embeddings_512 = None


def _get_embeddings_model():
    """Lazily instantiate the 512-dimension embedding model."""
    global _embeddings_512
    if _embeddings_512 is None:
        _embeddings_512 = get_embeddings(512)
    return _embeddings_512


def _normalize_query(query: str) -> str:
    """Normalize a query for consistent embedding generation.
    Strips case, punctuation, and collapses whitespace.
    """
    lowered = query.lower()
    no_punct = _re.sub(r"[^\w\s]", "", lowered)
    collapsed = _re.sub(r"\s+", " ", no_punct).strip()
    return collapsed


def _get_vector_client() -> redis.Redis:
    """Lazy-init a module-level sync Redis client for binary vectors (decode_responses=False)."""
    global _redis_vector_client
    if _redis_vector_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        # decode_responses must be False to safely store and retrieve binary vector buffers
        _redis_vector_client = redis.Redis.from_url(redis_url, decode_responses=False)
    return _redis_vector_client


def _ensure_index():
    """Lazily ensure the VSS search index exists in Redis."""
    global _index_ensured
    if _index_ensured:
        return
    client = _get_vector_client()
    index_name = "idx:faq_cache"
    try:
        client.ft(index_name).info()
        _index_ensured = True
    except Exception:
        # Index doesn't exist, create it
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
                f"Could not create RediSearch VSS index (Search module might be disabled): {e}"
            )


async def get_faq_response(customer_id: str, query: str) -> str | None:
    """
    Look up a cached FAQ response semantically in Redis.

    Uses OpenRouter/Gemini text embeddings to find nearest neighbors.
    Matches queries with cosine similarity >= 0.90 (cosine distance <= 0.10).

    Args:
        customer_id: Used for logging only.
        query: The raw user query string.

    Returns:
        The cached response string if a semantic hit exists, otherwise None.
    """
    try:
        client = _get_vector_client()
        _ensure_index()

        embeddings = _get_embeddings_model()
        normalized_query = _normalize_query(query)
        vector = await embeddings.aembed_query(normalized_query)
        # Pack float list into IEEE 754 float32 binary format
        vector_bytes = struct.pack(f"{len(vector)}f", *vector)

        # KNN Search for 1 nearest neighbor
        # Distance metric is COSINE, which returns (1 - CosineSimilarity)
        # Similarity threshold 0.90 corresponds to distance score <= 0.10
        index_name = "idx:faq_cache"
        q = (
            Query("*=>[KNN 1 @vector $vector_val AS score]")
            .sort_by("score")
            .return_fields("query", "response", "score")
            .dialect(2)
        )

        results = client.ft(index_name).search(
            q, query_params={"vector_val": vector_bytes}
        )

        if results.docs:
            doc = results.docs[0]
            score = float(doc.score)
            # Get threshold from environment (default similarity: 0.75 -> max distance: 0.25)
            threshold_similarity = float(os.getenv("FAQ_SEMANTIC_CACHE_THRESHOLD", "0.75"))
            max_distance = 1.0 - threshold_similarity

            if score <= max_distance:
                response_val = doc.response.decode("utf-8") if isinstance(doc.response, bytes) else str(doc.response)
                cached_query = doc.query.decode("utf-8") if isinstance(doc.query, bytes) else str(doc.query)
                logger.info(
                    f"🟢 FAQ SEMANTIC CACHE HIT | customer={customer_id} | score={score:.4f} (sim={1-score:.4f}, threshold={threshold_similarity:.2f}) | "
                    f"matched_query='{cached_query}' | query='{query}'"
                )
                return response_val
            else:
                closest_query = doc.query.decode("utf-8") if isinstance(doc.query, bytes) else str(doc.query)
                logger.info(
                    f"🔴 FAQ SEMANTIC CACHE MISS (Below threshold: {threshold_similarity:.2f}) | customer={customer_id} | closest_score={score:.4f} (sim={1-score:.4f}) | "
                    f"closest_query='{closest_query}' | query='{query}'"
                )
        else:
            logger.info(
                f"🔴 FAQ SEMANTIC CACHE MISS (No index hits) | customer={customer_id} | query='{query}'"
            )

        return None
    except redis.exceptions.ResponseError as e:
        if "No such index" in str(e):
            logger.warning(f"RediSearch Index missing: {e}. Semantic cache is disabled.")
        else:
            logger.warning(f"Redis ResponseError in FAQ cache lookup: {e}")
        return None
    except Exception as e:
        logger.warning(f"FAQ cache lookup failed (non-critical): {e}", exc_info=True)
        return None


async def set_faq_response(customer_id: str, query: str, response: str) -> None:
    """
    Store an FAQ response in the Redis VSS cache with a 5-minute TTL.

    Args:
        customer_id: Used for logging only.
        query: The raw user query string.
        response: The generated response to cache.
    """
    try:
        client = _get_vector_client()
        _ensure_index()

        embeddings = _get_embeddings_model()
        normalized_query = _normalize_query(query)
        vector = await embeddings.aembed_query(normalized_query)
        # Pack float list into IEEE 754 float32 binary format
        vector_bytes = struct.pack(f"{len(vector)}f", *vector)

        key = f"faq_resp:sem:{uuid.uuid4().hex}"

        # Write fields to Redis hash
        client.hset(
            key,
            mapping={
                "query": query,
                "response": response,
                "created_at": str(time.time()),
                "vector": vector_bytes,
            },
        )

        # Let Redis natively expire the key after 5 minutes (300 seconds)
        client.expire(key, _CACHE_TTL_SECONDS)
        logger.info(
            f"💾 FAQ response cached semantically | triggered_by={customer_id} | key={key} | query='{query}' | TTL={_CACHE_TTL_SECONDS}s"
        )
    except redis.exceptions.ResponseError as e:
        if "No such index" in str(e):
            logger.warning(f"RediSearch Index missing: {e}. Cannot write to semantic cache.")
        else:
            logger.warning(f"Redis ResponseError in FAQ cache write: {e}")
    except Exception as e:
        logger.warning(f"FAQ cache write failed (non-critical): {e}", exc_info=True)
