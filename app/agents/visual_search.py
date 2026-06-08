from typing import Any
from pydantic import BaseModel, Field
from app.db.qdrant import get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, Range
from app.services.llm import get_llm
from app.services.clip_embedder import embed_text
from langchain_core.messages import SystemMessage, HumanMessage
import json
import logging

logger = logging.getLogger(__name__)

class RerankedResult(BaseModel):
    product_id: str = Field(description="The ID of the product")
    relevance_score: float = Field(description="Score from 0.0 to 1.0 on how well it matches the user constraints")
    reasoning: str = Field(description="Why it matches or fails")

class RerankResponse(BaseModel):
    results: list[RerankedResult]

def build_filters(active_filters: dict | None):
    if not active_filters:
        return None
    conditions = []
    if "max_price" in active_filters:
        conditions.append(FieldCondition(key="price", range=Range(lte=active_filters["max_price"])))
    if "min_price" in active_filters:
        conditions.append(FieldCondition(key="price", range=Range(gte=active_filters["min_price"])))
    
    return Filter(must=conditions) if conditions else None

async def visual_search_node(state: Any) -> dict:
    """Search Qdrant product_images collection with CLIP vector and hybrid LLM reranking."""
    
    qdrant = get_qdrant_client()
    filters = build_filters(state.get("active_filters", {}))
    search_query = state.get("search_query")
    image_embedding = state.get("image_embedding")
    image_description = state.get("image_description") # OCR from image_analyzer
    
    # 1. Determine primary vector
    query_vector = None
    if image_embedding:
        query_vector = image_embedding
    elif search_query:
        try:
            query_vector = embed_text(search_query)
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")

    if not query_vector:
        return {"visual_results": []}

    ranked_dict = {}

    try:
        # Vector Search
        response = await qdrant.query_points(
            collection_name="product_images",
            query=query_vector,
            query_filter=filters,
            limit=5, # Reduced from 10 to prevent LLM output token truncation during reranking
            score_threshold=0.20,
            with_payload=True
        )
        
        candidates = []
        for r in response.points:
            pid = str(r.id)
            payload = r.payload or {}
            item = {**payload, "score": round(r.score, 4)}
            ranked_dict[pid] = item
            candidates.append(item)

        # 2. LLM Reranking / Filtering (only if text constraints exist)
        if candidates and (search_query or image_description):
            llm = get_llm(temperature=0.0).with_structured_output(RerankResponse)
            
            prompt = (
                "You are an AI ranking assistant. You have retrieved a list of visually similar products.\n"
                "Your job is to filter and rerank them based on the user's specific text constraints.\n\n"
            )
            if search_query:
                prompt += f"User Constraint / Text Query: \"{search_query}\"\n"
            if image_description:
                prompt += f"Text extracted from image (OCR): \"{image_description}\"\n"
                
            prompt += "\nEvaluate each product below. If it violates the user's explicit text constraint, give it a score of 0.0. "
            prompt += "If it perfectly matches, give it 1.0. If it partially matches, give it 0.5.\n\n"
            prompt += "Products:\n"
            for c in candidates:
                prompt += f"- ID: {c.get('product_id')} | Title: {c.get('title')} | Category: {c.get('category')} | Desc: {c.get('description')[:100]}...\n"

            try:
                llm_resp = await llm.ainvoke([SystemMessage(content=prompt)])
                
                # Apply new scores
                for res in llm_resp.results:
                    pid = str(res.product_id)
                    if pid in ranked_dict:
                        ranked_dict[pid]["relevance_score"] = res.relevance_score
                        ranked_dict[pid]["relevance_reasoning"] = res.reasoning
                
                # Filter out those with 0.0 relevance and sort
                final_list = [v for v in ranked_dict.values() if v.get("relevance_score", 1.0) > 0.0]
                final_list.sort(key=lambda x: (x.get("relevance_score", 0), x.get("score", 0)), reverse=True)
                
            except Exception as e:
                logger.error(f"LLM Reranking failed: {e}")
                final_list = candidates
        else:
            final_list = candidates

        return {"visual_results": final_list[:5]}
        
    except Exception as e:
        logger.exception(f"Visual search error: {e}")
        return {"visual_results": []}
