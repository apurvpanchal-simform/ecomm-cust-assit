from typing import Any
from langchain_core.messages import AIMessage
from langfuse import observe
import logging
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
    """

    search_query = state.get("search_query")
    image_embedding = state.get("image_embedding")
    image_description = state.get("image_description")  # OCR from image_analyzer
    image_tags = state.get("image_tags", [])
    active_filters = state.get("active_filters", {})
    has_real_image = bool(state.get("image_base64"))

    logger.info(
        f"🔍 Image Search Request -> Query: '{search_query}', Filters: {active_filters}, Has Image: {has_real_image}"
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
        msg += "| Image | Product | Price |\n"
        msg += "| :---: | :--- | :--- |\n"
        for idx, item in enumerate(final_results):
            title = item.get("title", f"Product {idx+1}").replace("|", "-")
            price = item.get("price", "N/A")
            image_url = item.get("image_url", "")
            category = item.get("category", "").replace("|", "-")

            img_html = (
                f'<img src="{image_url}" width="120" height="120" style="object-fit: cover; border-radius: 8px;">'
                if image_url
                else ""
            )

            info_html = f"**{title}**"
            if category:
                info_html += f"<br>_{category.title()}_"

            msg += f"| {img_html} | {info_html} | **Rs. {price}** |\n"

    new_messages = [AIMessage(content=msg, name="image_search_agent")]

    return {
        "image_results": final_results,
        "messages": new_messages,
        "executed_agents": state.get("executed_agents", []) + ["image_search_agent"],
    }
