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

from app.db.qdrant import get_qdrant_client
from app.services.dense_embedder import embed_text
from app.services.sparse_embedder import embed_sparse_text

logger = logging.getLogger(__name__)


def build_filters(active_filters: dict | None):
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


class ImageRetriever:
    def __init__(self):
        self.qdrant = get_qdrant_client()

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
        Executes a multimodal (Dense + Sparse) Hybrid Search against Qdrant,
        followed by LLM reranking of the top results. Text search_query is always required.
        """
        filters = build_filters(active_filters)

        # 1. Determine Vectors for Multi-Vector Hybrid Search
        image_vector = image_embedding
        text_vector = None
        sparse_vector = None

        try:
            text_vector = embed_text(search_query)
            if image_tags:
                tags_text = " ".join(image_tags)
                combined_text = f"{search_query} {tags_text}"
                logger.info(
                    f"Using Image Tags + Query for Sparse Keyword Search: {combined_text}"
                )
                sparse_vector = embed_sparse_text(combined_text)
            else:
                sparse_vector = embed_sparse_text(search_query)
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")

        ranked_dict = {}

        try:
            # 2. Build Prefetch Queries for Dense-Dense Hybrid Search
            prefetch_queries = []

            # Dense Image Prefetch (Only if there is a real image uploaded)
            if image_vector and has_real_image:
                prefetch_queries.append(
                    Prefetch(
                        query=image_vector,
                        using="",
                        limit=20,
                        filter=filters,
                        score_threshold=0.18,  # Lower threshold for image-to-image
                    )
                )

            # Dense Text Prefetch (Always search the text intent)
            if text_vector:
                prefetch_queries.append(
                    Prefetch(
                        query=text_vector,
                        using="",
                        limit=20,
                        filter=filters,
                        score_threshold=0.22,  # Higher threshold for text-to-image
                    )
                )

            # Sparse Text Prefetch (BM25)
            if sparse_vector and sparse_vector.get("indices"):
                prefetch_queries.append(
                    Prefetch(
                        query=SparseVector(**sparse_vector),
                        using="text",
                        limit=20,
                        filter=filters,
                    )
                )

            if not prefetch_queries:
                logger.info(
                    "⚠️ No valid prefetch queries generated. Returning empty results."
                )
                return []

            # 3. Explicitly log individual vector results for observability
            if image_vector and has_real_image:
                try:
                    res_img = await self.qdrant.query_points(
                        collection_name="product_images",
                        query=image_vector,
                        using="",
                        limit=5,
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
                    logger.warning(f"Failed to log image results: {e}")

            if text_vector:
                try:
                    res_txt = await self.qdrant.query_points(
                        collection_name="product_images",
                        query=text_vector,
                        using="",
                        limit=5,
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
                    logger.warning(f"Failed to log text results: {e}")

            if sparse_vector and sparse_vector.get("indices"):
                try:
                    res_sparse = await self.qdrant.query_points(
                        collection_name="product_images",
                        query=SparseVector(**sparse_vector),
                        using="text",
                        limit=5,
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
                    logger.warning(f"Failed to log sparse results: {e}")

            # 4. Execute Qdrant Query Fusion
            logger.info(
                f"🚀 Executing Qdrant Multi-Vector RRF Fusion Search with {len(prefetch_queries)} vector modalities."
            )

            response = await self.qdrant.query_points(
                collection_name="product_images",
                prefetch=prefetch_queries,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=3,  # Only return the top 3 results from the fusion
                with_payload=True,
            )

            candidates = []
            for r in response.points:
                pid = str(r.id)
                payload = r.payload or {}
                item = {**payload, "score": round(r.score, 4)}
                ranked_dict[pid] = item
                candidates.append(item)

            log_msg = "\n========== RAW QDRANT RESULTS (PRE-RERANK) ==========\n"
            if not candidates:
                log_msg += "No results found.\n"
            for idx, c in enumerate(candidates, 1):
                log_msg += f"{idx}. ID: {c.get('product_id')} | Title: {c.get('title')} | Score: {c.get('score')}\n"
            log_msg += "=======================================================\n"
            logger.info(log_msg)

            # We bypass LLM reranking entirely as requested
            return candidates

        except Exception as e:
            logger.exception(f"Image search error: {e}")
            return []
