"""
Node for generating dense vector embeddings using the SigLIP2 model.
Handles both text and image modality embeddings for visual search.
"""

import asyncio
import logging
from typing import Any

# from app.services.vision_service import vectorize_image_base64, vectorize_text
from langfuse import observe

from app.services.dense_embedder import embed_image_base64, embed_text

logger = logging.getLogger(__name__)


@observe(name="clip_embedding_node")
async def clip_embedding_node(state: Any) -> dict:
    """
    Generate dense vector embeddings using SigLIP2 for image search.

    If an image is uploaded (`image_base64`), it generates an image embedding.
    If no image is uploaded, it checks for a text `search_query` and generates
    a text embedding that aligns with the visual semantic space.

    Args:
        state: The global agent state.

    Returns:
        A dictionary containing the generated `image_embedding`.
    """
    # If there's an image, we use image to search
    if state.get("image_base64"):
        logger.info("🖼️ Uploaded image detected. Routing to SigLIP Image Encoder...")
        vector = await asyncio.to_thread(embed_image_base64, state["image_base64"])
        logger.info("✅ Generated Dense Image Vector (dim: %s)", len(vector))
        return {"image_embedding": vector}

    # If no image, we check if there's a search_query to search visually
    user_text = state.get("search_query")

    if not user_text:
        return {}

    # Embed the text using SigLIP2 to search for images matching the text
    logger.info(
        logger.info(
            "📝 Text-only query detected ('%s'). Routing to SigLIP Text Encoder...",
            user_text,
        )
    )
    vector = await asyncio.to_thread(embed_text, user_text)
    logger.info("✅ Generated Dense Text Vector (dim: %s)", len(vector))
    return {"image_embedding": vector}
