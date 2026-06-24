"""
Agent node that performs multimodal product discovery using Hybrid Search.

`image_search_node` is the final step in the image search pipeline:
  clip_embedder → image_search → cleanup

It delegates the actual search to `ImageRetriever.search_and_rerank()`, which
runs a multi-vector Qdrant Hybrid Search (Dense Image + Dense Text + Sparse BM25)
followed by Reciprocal Rank Fusion (RRF).  The top results are formatted into
a markdown message for the user.
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
    Execute a multimodal Hybrid Search and format the top results as a markdown message.

    Reads embeddings, query text, image metadata, and price filters from the state,
    delegates to `ImageRetriever.search_and_rerank()`, and builds a user-facing
    product listing.

    Args:
        state: The global AgentState containing:
            search_query      — text description of the desired product
            image_embedding   — pre-computed SigLIP2 vector (from clip_embedder)
            image_description — Azure CV caption / OCR text
            image_tags        — Azure CV object tags
            active_filters    — optional price range filters
            image_base64      — used to determine if a real image was uploaded

    Returns:
        A dict with:
            image_results    — list of raw product dicts returned by Qdrant
            messages         — list containing the formatted AIMessage
            executed_agents  — updated list marking 'image_search_agent' as done
    """
    # ── 1. Extract inputs from state ──────────────────────────────────────────
    search_query = state.get("search_query")
    image_embedding = state.get("image_embedding")
    image_description = state.get("image_description")  # OCR from image_analyzer
    image_tags = state.get("image_tags", [])
    active_filters = state.get("active_filters", {})
    has_real_image = bool(state.get("image_base64"))

    logger.info(
        "🔍 Image Search Request -> Query: '%s', Filters: %s, Has Image: %s",
        search_query,
        active_filters,
        has_real_image,
    )

    # ── 2. Run Hybrid Search + RRF Fusion ─────────────────────────────────────
    retriever = ImageRetriever()
    final_results = await retriever.search_and_rerank(
        search_query=search_query,
        image_embedding=image_embedding,
        image_description=image_description,
        image_tags=image_tags,
        active_filters=active_filters,
        has_real_image=has_real_image,
    )

    # ── 3. Format results into a user-facing markdown message ─────────────────
    if not final_results:
        query_hint = f' for **"{search_query}"**' if search_query else ""
        msg = (
            f"I couldn't find any matching products{query_hint} in our catalog. 😔\n\n"
            "Here are a few things you can try:\n"
            "- **Broaden your search** — try a more general term (e.g. *\"jacket\"* instead of *\"white jacket\"*)\n"
            "- **Upload a photo** — if you have a reference image, I can use it to find visually similar products\n"
            "- **Adjust filters** — if you set a price range, try widening it\n\n"
            "Would you like me to help you search for something else?"
        )
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
