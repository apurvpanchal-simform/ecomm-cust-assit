import asyncio
from typing import Any
from app.services.dense_embedder import embed_image_base64, embed_text

# from app.services.vision_service import vectorize_image_base64, vectorize_text
from langfuse import observe
import logging

logger = logging.getLogger(__name__)


@observe(name="clip_embedding_node")
async def clip_embedding_node(state: Any) -> dict:
    """Generate SigLIP2 embedding — pure image OR text-only."""
    # If there's an image, we use image to search
    if state.get("image_base64"):
        logger.info("🖼️ Uploaded image detected. Routing to SigLIP Image Encoder...")
        vector = await asyncio.to_thread(embed_image_base64, state["image_base64"])
        logger.info(f"✅ Generated Dense Image Vector (dim: {len(vector)})")
        return {"image_embedding": vector}

    # If no image, we check if there's a search_query to search visually
    user_text = state.get("search_query")

    if not user_text:
        return {}

    # Embed the text using SigLIP2 to search for images matching the text
    logger.info(f"📝 Text-only query detected ('{user_text}'). Routing to SigLIP Text Encoder...")
    vector = await asyncio.to_thread(embed_text, user_text)
    logger.info(f"✅ Generated Dense Text Vector (dim: {len(vector)})")
    return {"image_embedding": vector}
