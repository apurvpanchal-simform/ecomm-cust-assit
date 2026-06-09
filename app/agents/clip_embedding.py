import asyncio
from typing import Any
from app.services.clip_embedder import embed_image_base64, embed_text
from langsmith import traceable

@traceable(name="clip_embedding_node")
async def clip_embedding_node(state: Any) -> dict:
    """Generate CLIP embedding — pure image OR text-only."""
    # If there's an image, we use image to search
    if state.get("image_base64"):
        vector = await asyncio.to_thread(embed_image_base64, state["image_base64"])
        return {"image_embedding": vector}
        
    # If no image, we check if there's a search_query to search visually
    user_text = state.get("search_query")
    
    if not user_text:
        return {}
        
    # Embed the text using CLIP to search for images matching the text
    vector = await asyncio.to_thread(embed_text, user_text)
    return {"image_embedding": vector}
