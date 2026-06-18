"""
Agent node that performs multi-modal product discovery using Hybrid Search and LLM reranking.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langfuse import observe

from app.rag.image_retriever import ImageRetriever

logger = logging.getLogger(__name__)


@observe(name="image_search_node")
async def image_search_node(state: Any) -> dict:
    """
    Executes a multimodal (Dense + Sparse) Hybrid Search against Qdrant, followed by LLM reranking.

    This node acts as the visual and semantic product discovery specialist. When the supervisor
    identifies a product search intent (via text, an uploaded image, or both), it routes to this node.
    The node performs a multi-vector fusion query and then re-ranks the raw vector search results
    using an LLM to ensure absolute relevance to the user's nuanced intent.

    Args:
        state: The global AgentState containing embeddings, extracted text, and filters.

    Returns:
        A dictionary containing the `image_results` list, the generated `messages` array
        with the markdown table, and the updated `executed_agents` list.
    """
    search_query = state.get("search_query")
    image_embedding = state.get("image_embedding")
    image_description = state.get("image_description")  # OCR from image_analyzer
    image_tags = state.get("image_tags", [])
    active_filters = state.get("active_filters", {})
    has_real_image = bool(state.get("image_base64"))

    logger.info(
        logger.info(
            "🔍 Image Search Request -> Query: '%s', Filters: %s, Has Image: %s",
            search_query,
            active_filters,
            has_real_image,
        )
    )

    retriever = ImageRetriever()
    final_results = await retriever.search_and_rerank(
        search_query=search_query,
        image_embedding=image_embedding,
        image_description=image_description,
        image_tags=image_tags,
        active_filters=active_filters,
        has_real_image=has_real_image,
    )

    if not final_results:
        msg = "I couldn't find any products matching your search."
    else:
        msg = "Here are the top matches I found:\n\n"
        for idx, item in enumerate(final_results):
            title = item.get("title", f"Product {idx + 1}")
            price = item.get("price", "N/A")
            image_url = item.get("image_url", "")
            category = item.get("category", "")

            msg += f"**{title}**\n\n"
            if category:
                msg += f"_{category.title()}_\n\n"
            msg += f"**Rs. {price}**\n\n"
            if image_url:
                msg += f"![{title}]({image_url})\n\n"
            msg += "---\n\n"

    new_messages = [AIMessage(content=msg)]

    return {
        "image_results": final_results,
        "messages": new_messages,
        "executed_agents": state.get("executed_agents", []) + ["image_search_agent"],
    }
