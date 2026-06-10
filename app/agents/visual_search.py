from typing import Any
from app.db.qdrant import get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, Range, Prefetch, FusionQuery, Fusion
from app.services.llm import get_llm
from app.services.clip_embedder import embed_text
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langsmith import traceable
from app.schemas.search import RerankedResult, RerankResponse
import json
import logging

logger = logging.getLogger(__name__)

def build_filters(active_filters: dict | None):
    if not active_filters:
        return None
    conditions = []
    if "max_price" in active_filters:
        conditions.append(FieldCondition(key="price", range=Range(lte=active_filters["max_price"])))
    if "min_price" in active_filters:
        conditions.append(FieldCondition(key="price", range=Range(gte=active_filters["min_price"])))
    
    return Filter(must=conditions) if conditions else None

@traceable(name="visual_search_node")
async def visual_search_node(state: Any) -> dict:
    """Search Qdrant product_images collection using Hybrid Search (Dense + Sparse) with LLM reranking."""
    
    qdrant = get_qdrant_client()
    filters = build_filters(state.get("active_filters", {}))
    search_query = state.get("search_query")
    image_embedding = state.get("image_embedding")
    image_description = state.get("image_description") # OCR from image_analyzer
    
    # 1. Determine Vectors for Multi-Vector Hybrid Search
    image_vector = image_embedding
    text_vector = None
    if search_query:
        try:
            text_vector = embed_text(search_query)
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")

    ranked_dict = {}

    try:
        # 2. Build Prefetch Queries for Dense-Dense Hybrid Search
        prefetch_queries = []
        
        # Dense Image Prefetch
        if image_vector:
            prefetch_queries.append(
                Prefetch(
                    query=image_vector,
                    using="",
                    limit=20,
                    filter=filters,
                    score_threshold=0.20
                )
            )
            
        # Dense Text Prefetch
        if text_vector:
            prefetch_queries.append(
                Prefetch(
                    query=text_vector,
                    using="",
                    limit=20,
                    filter=filters,
                    score_threshold=0.20
                )
            )

        if not prefetch_queries:
            return {"visual_results": []}

        # 3. Execute Qdrant Query
        if len(prefetch_queries) == 1:
            response = await qdrant.query_points(
                collection_name="product_images",
                query=prefetch_queries[0].query,
                using=prefetch_queries[0].using,
                query_filter=filters,
                limit=5,
                with_payload=True
            )
        else:
            response = await qdrant.query_points(
                collection_name="product_images",
                prefetch=prefetch_queries,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=5,
                with_payload=True
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

        final_results = final_list[:5]
        
    except Exception as e:
        logger.exception(f"Visual search error: {e}")
        final_results = []
        
    if not final_results:
        msg = "I couldn't find any products matching your search."
    else:
        msg = "Here are the top matches I found:\n\n"
        for idx, item in enumerate(final_results):
            title = item.get("title", f"Product {idx+1}")
            price = item.get("price", "N/A")
            image_url = item.get("image_url", "")
            
            msg += f"**{title}** - ${price}\n"
            if image_url:
                msg += f'<img src="{image_url}" width="150" style="border-radius: 8px; margin-top: 5px;">\n\n'
            else:
                msg += "\n"

    new_messages = [AIMessage(content=msg, name="visual_search_agent")]

    return {
        "visual_results": final_results,
        "messages": new_messages,
        "executed_agents": state.get("executed_agents", []) + ["visual_search_agent"]
    }
