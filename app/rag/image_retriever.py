"""
Multimodal product image retriever using Qdrant Hybrid Search (Dense + Sparse).

`ImageRetriever.search_and_rerank()` is the main entry point.  It:
1. Computes dense text and/or image vectors plus a sparse BM25 vector.
2. Builds a set of Qdrant Prefetch queries (one per modality).
3. Fires individual debug queries for observability logging.
4. Executes a Reciprocal Rank Fusion (RRF) query to merge all modalities.
5. Returns the top-N ranked candidates.

LLM re-ranking has been disabled in favour of direct RRF output.
"""

import asyncio
import logging

from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    Prefetch,
    Range,
    SparseVector,
)

from app.config import SEARCH_CONFIG
from app.db.qdrant import get_qdrant_client
from app.services.dense_embedder import embed_text
from app.services.sparse_embedder import embed_sparse_text

logger = logging.getLogger(__name__)


# ── Filter builder ────────────────────────────────────────────────────────────


def build_filters(active_filters: dict | None) -> Filter | None:
    """
    Convert an `active_filters` dict into a Qdrant `Filter` object.

    Currently supports `min_price` and `max_price` range conditions.
    Returns None if no filters are provided (Qdrant skips filtering entirely).

    Args:
        active_filters: Dict optionally containing 'min_price' and/or 'max_price'.

    Returns:
        A Qdrant Filter with `must` conditions, or None if no filters apply.
    """
    if not active_filters:
        return None

    conditions = []
    if "max_price" in active_filters:
        conditions.append(
            FieldCondition(key="price", range=Range(lte=active_filters["max_price"]))
        )
    if "min_price" in active_filters:
        conditions.append(
            FieldCondition(key="price", range=Range(gte=active_filters["min_price"]))
        )

    return Filter(must=conditions) if conditions else None


# ── ImageRetriever ────────────────────────────────────────────────────────────


class ImageRetriever:
    """
    Retrieves product images from Qdrant using multimodal Hybrid Search.

    Supports three search modalities that can be combined:
    - Dense image vector (SigLIP2 image encoder) — used when a real image is uploaded.
    - Dense text vector  (SigLIP2 text encoder)  — always used for the text query.
    - Sparse text vector (BM25 via SPLADE)        — used for keyword matching.

    All three Prefetch results are merged using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self):
        self.qdrant = get_qdrant_client()

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _compute_vectors(
        self,
        search_query: str | None,
        image_tags: list[str] | None,
    ) -> tuple[list[float] | None, dict | None]:
        """
        Compute dense text and sparse BM25 vectors for the given query.

        Both `embed_text` (PyTorch) and `embed_sparse_text` (SPLADE CPU model)
        are synchronous blocking functions.  They are run via `asyncio.to_thread`
        so that they do not block the event loop and drop WebSocket heartbeats.

        Args:
            search_query: The text query string, which can be None or empty.
            image_tags: Optional tags extracted from an uploaded image;
                        merged with the query for better sparse matching.

        Returns:
            A tuple of (text_vector, sparse_vector_dict) where either may be None
            if embedding failed.
        """
        text_vector = None
        sparse_vector = None

        try:
            query_str = search_query or ""
            if query_str:
                # embed_text is CPU-bound (PyTorch) — run in thread pool
                text_vector = await asyncio.to_thread(embed_text, query_str)

            if image_tags:
                # Combine query + tags for richer keyword matching
                tags_text = " ".join(image_tags)
                combined_text = f"{query_str} {tags_text}".strip()
                if combined_text:
                    # embed_sparse_text is CPU-bound (SPLADE) — run in thread pool
                    sparse_vector = await asyncio.to_thread(
                        embed_sparse_text, combined_text
                    )
            elif query_str:
                sparse_vector = await asyncio.to_thread(embed_sparse_text, query_str)

        except Exception as e:
            logger.error("Failed to embed text: %s", e)

        return text_vector, sparse_vector

    def _build_prefetch_queries(
        self,
        image_vector: list[float] | None,
        text_vector: list[float] | None,
        sparse_vector: dict | None,
        has_real_image: bool,
        filters: Filter | None,
    ) -> list[Prefetch]:
        """
        Assemble the list of Qdrant Prefetch objects for the multi-vector fusion query.

        Each Prefetch runs a candidate retrieval for one modality.  RRF then merges
        all candidates into a unified ranked list.

        Args:
            image_vector: Dense SigLIP2 image embedding (only used when has_real_image=True).
            text_vector: Dense SigLIP2 text embedding.
            sparse_vector: BM25 sparse vector dict with 'indices' and 'values'.
            has_real_image: Whether the user uploaded a real image (not just text).
            filters: Optional Qdrant price range filter.

        Returns:
            List of Prefetch objects (may be empty if all embeddings failed).
        """
        prefetch_queries = []

        # Dense image Prefetch — only when a real image was uploaded
        if image_vector and has_real_image:
            prefetch_queries.append(
                Prefetch(
                    query=image_vector,
                    using="",  # Default (unnamed) dense vector namespace
                    limit=SEARCH_CONFIG.prefetch_limit,
                    filter=filters,
                    score_threshold=SEARCH_CONFIG.image_score_threshold,
                )
            )

        # Dense text Prefetch — always included for intent matching
        if text_vector:
            prefetch_queries.append(
                Prefetch(
                    query=text_vector,
                    using="",
                    limit=SEARCH_CONFIG.prefetch_limit,
                    filter=filters,
                    score_threshold=SEARCH_CONFIG.text_score_threshold,
                )
            )

        # Sparse BM25 Prefetch — keyword matching on product titles/descriptions
        if sparse_vector and sparse_vector.get("indices"):
            prefetch_queries.append(
                Prefetch(
                    query=SparseVector(**sparse_vector),
                    using="text",  # Named sparse vector namespace in Qdrant
                    limit=SEARCH_CONFIG.prefetch_limit,
                    filter=filters,
                )
            )

        return prefetch_queries

    async def _log_individual_vector_results(
        self,
        image_vector: list[float] | None,
        text_vector: list[float] | None,
        sparse_vector: dict | None,
        has_real_image: bool,
        filters: Filter | None,
    ) -> None:
        """
        Run and log individual per-modality searches for debugging/observability.

        These queries run in addition to the main fusion query and their results
        are only logged, not returned to the caller.

        Args:
            image_vector, text_vector, sparse_vector: Computed embedding vectors.
            has_real_image: Whether a real image is present.
            filters: Optional price filter.
        """
        # Dense image results (only when a real image was uploaded)
        if image_vector and has_real_image:
            try:
                res_img = await self.qdrant.query_points(
                    collection_name="product_images",
                    query=image_vector,
                    using="",
                    limit=SEARCH_CONFIG.log_limit,
                    query_filter=filters,
                    with_payload=True,
                )
                log_msg = "\n=== 🖼️ DENSE IMAGE RESULTS ===\n"
                if not res_img.points:
                    log_msg += "No image results found.\n"
                for idx, p in enumerate(res_img.points, 1):
                    log_msg += (
                        f"{idx}. {p.payload.get('title')} (score: {p.score:.4f})\n"
                    )
                logger.info(log_msg)
            except Exception as e:
                logger.warning("Failed to log image results: %s", e)

        # Dense text results
        if text_vector:
            try:
                res_txt = await self.qdrant.query_points(
                    collection_name="product_images",
                    query=text_vector,
                    using="",
                    limit=SEARCH_CONFIG.log_limit,
                    query_filter=filters,
                    with_payload=True,
                )
                log_msg = "\n=== 📝 DENSE TEXT RESULTS ===\n"
                if not res_txt.points:
                    log_msg += "No dense text results found.\n"
                for idx, p in enumerate(res_txt.points, 1):
                    log_msg += (
                        f"{idx}. {p.payload.get('title')} (score: {p.score:.4f})\n"
                    )
                logger.info(log_msg)
            except Exception as e:
                logger.warning("Failed to log text results: %s", e)

        # Sparse BM25 results
        if sparse_vector and sparse_vector.get("indices"):
            try:
                res_sparse = await self.qdrant.query_points(
                    collection_name="product_images",
                    query=SparseVector(**sparse_vector),
                    using="text",
                    limit=SEARCH_CONFIG.log_limit,
                    query_filter=filters,
                    with_payload=True,
                )
                log_msg = "\n=== 🔑 SPARSE TEXT (BM25) RESULTS ===\n"
                if not res_sparse.points:
                    log_msg += "No sparse results found.\n"
                for idx, p in enumerate(res_sparse.points, 1):
                    log_msg += (
                        f"{idx}. {p.payload.get('title')} (score: {p.score:.4f})\n"
                    )
                logger.info(log_msg)
            except Exception as e:
                logger.warning("Failed to log sparse results: %s", e)

    async def _execute_fusion_search(
        self,
        prefetch_queries: list[Prefetch],
    ) -> list[dict]:
        """
        Execute the Qdrant RRF Fusion query and return the top-3 candidates.

        Merges all Prefetch candidate sets using Reciprocal Rank Fusion (RRF),
        which rewards items that rank highly in multiple modalities.

        Args:
            prefetch_queries: List of Prefetch objects built by `_build_prefetch_queries`.

        Returns:
            A list of product payload dicts with an added 'score' field,
            or an empty list if no results are found.
        """
        logger.info(
            "🚀 Executing Qdrant Multi-Vector RRF Fusion Search with %s vector modalities.",
            len(prefetch_queries),
        )

        response = await self.qdrant.query_points(
            collection_name="product_images",
            prefetch=prefetch_queries,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=SEARCH_CONFIG.fusion_limit,
            with_payload=True,
        )

        candidates = []
        for r in response.points:
            payload = r.payload or {}
            item = {**payload, "score": round(r.score, 4)}
            candidates.append(item)

        # Log pre-rerank results for observability
        log_msg = "\n========== RAW QDRANT RESULTS (PRE-RERANK) ==========\n"
        if not candidates:
            log_msg += "No results found.\n"
        for idx, c in enumerate(candidates, 1):
            log_msg += (
                f"{idx}. ID: {c.get('product_id')} | "
                f"Title: {c.get('title')} | "
                f"Score: {c.get('score')}\n"
            )
        log_msg += "=======================================================\n"
        logger.info(log_msg)

        return candidates

    # ── Public entry point ────────────────────────────────────────────────────

    async def search_and_rerank(
        self,
        search_query: str,
        image_embedding: list[float] | None,
        image_description: str | None,
        image_tags: list[str] | None,
        active_filters: dict | None,
        has_real_image: bool = False,
    ) -> list[dict]:
        """
        Run multimodal Hybrid Search and return ranked product candidates.

        Orchestrates the full pipeline:
        1. Build price filters.
        2. Compute text and sparse vectors.
        3. Build Prefetch queries for each available modality.
        4. Log individual vector results for debugging.
        5. Execute RRF fusion search.

        Args:
            search_query: Text description of the desired product.
            image_embedding: Pre-computed SigLIP2 image vector (from clip_embedding_node),
                             or None if no image was uploaded.
            image_description: Azure CV caption/OCR text from the uploaded image.
            image_tags: Azure CV tag list from the uploaded image.
            active_filters: Dict with optional 'min_price' and/or 'max_price'.
            has_real_image: True if the user uploaded an actual image file.

        Returns:
            A list of product payload dicts sorted by RRF score (descending),
            or an empty list if the search fails or returns no results.
        """
        # ── 1. Build Qdrant price filters ─────────────────────────────────────
        filters = build_filters(active_filters)

        # The image vector was computed upstream by clip_embedding_node
        image_vector = image_embedding

        # ── 2. Compute text and sparse vectors (async — uses thread pool) ──────
        text_vector, sparse_vector = await self._compute_vectors(search_query, image_tags)

        # ── 3. Build Prefetch queries per modality ────────────────────────────
        prefetch_queries = self._build_prefetch_queries(
            image_vector=image_vector,
            text_vector=text_vector,
            sparse_vector=sparse_vector,
            has_real_image=has_real_image,
            filters=filters,
        )

        if not prefetch_queries:
            logger.info(
                "⚠️ No valid prefetch queries generated. Returning empty results."
            )
            return []

        try:
            # ── 4. Log individual modality results for debugging ──────────────
            await self._log_individual_vector_results(
                image_vector=image_vector,
                text_vector=text_vector,
                sparse_vector=sparse_vector,
                has_real_image=has_real_image,
                filters=filters,
            )

            # ── 5. Execute fusion search and return results ───────────────────
            return await self._execute_fusion_search(prefetch_queries)

        except Exception as e:
            logger.exception("Image search error: %s", e)
            return []
