"""
Graph node for generating dense SigLIP2 vector embeddings for visual search.

This node runs as part of the image search pipeline (after the supervisor routes
to `image_search_agent`).  It produces a single `image_embedding` vector that
the downstream `image_search_node` uses for Qdrant similarity search.

Embedding strategy
------------------
- If a real image was uploaded (`image_base64` is set): encode the image directly.
  This gives the best visual similarity — pixels-to-pixels.
- If only a text query is present (`search_query` is set): encode the text into
  the same SigLIP2 visual-semantic space so text → image search still works.
- If neither is set: return an empty dict (the fusion search will use sparse-only).
"""

import asyncio
import logging
from typing import Any

from langfuse import observe

from app.services.dense_embedder import embed_image_base64, embed_text

logger = logging.getLogger(__name__)


@observe(name="clip_embedding_node")
async def clip_embedding_node(state: Any) -> dict:
    """
    Generate a dense SigLIP2 vector embedding for image or text-based product search.

    Runs `embed_image_base64` or `embed_text` in a thread pool executor
    (`asyncio.to_thread`) to avoid blocking the event loop with the CPU-bound
    PyTorch inference.

    Args:
        state: The global AgentState containing `image_base64` and/or `search_query`.

    Returns:
        A dict with `image_embedding` (list[float]) if embedding succeeded,
        or an empty dict if neither an image nor a search query is available.
    """
    # ── Case 1: User uploaded an image — use image encoder ───────────────────
    if state.get("image_base64"):
        logger.info("🖼️ Uploaded image detected. Routing to SigLIP Image Encoder...")
        vector = await asyncio.to_thread(embed_image_base64, state["image_base64"])
        logger.info("✅ Generated Dense Image Vector (dim: %s)", len(vector))
        return {"image_embedding": vector}

    # ── Case 2: Text-only query — use text encoder in the visual space ────────
    user_text = state.get("search_query")
    if not user_text:
        # No image and no search query — nothing to embed
        return {}

    logger.info(
        "📝 Text-only query detected ('%s'). Routing to SigLIP Text Encoder...",
        user_text,
    )
    vector = await asyncio.to_thread(embed_text, user_text)
    logger.info("✅ Generated Dense Text Vector (dim: %s)", len(vector))
    return {"image_embedding": vector}
