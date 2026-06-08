from typing import Any
from app.db.qdrant import get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, Range

def build_filters(active_filters: dict | None):
    if not active_filters:
        return None
    conditions = []
    # We only have price filter now based on our simplified schema
    if "max_price" in active_filters:
        conditions.append(FieldCondition(key="price", range=Range(lte=active_filters["max_price"])))
    if "min_price" in active_filters:
        conditions.append(FieldCondition(key="price", range=Range(gte=active_filters["min_price"])))
    
    return Filter(must=conditions) if conditions else None

async def visual_search_node(state: Any) -> dict:
    """Search Qdrant product_images collection with CLIP vector."""
    if not state.get("image_embedding"):
        return {}

    qdrant = get_qdrant_client()
    
    # Build optional filters from conversation context
    filters = build_filters(state.get("active_filters", {}))

    try:
        response = await qdrant.query_points(
            collection_name="product_images",
            query=state["image_embedding"],
            query_filter=filters,
            limit=3,
            score_threshold=0.25, # Lowered threshold because CLIP text-image cross modal scores can be lower
            with_payload=True
        )
        
        ranked = []
        for r in response.points:
            ranked.append({**r.payload, "score": round(r.score, 4)})
            
        return {"visual_results": ranked}
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception(f"Visual search error: {e}")
        return {"visual_results": []}
