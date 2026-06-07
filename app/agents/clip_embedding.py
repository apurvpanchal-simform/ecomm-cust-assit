import asyncio
from typing import Any
from app.services.clip_embedder import embed_image_base64, embed_text

async def clip_embedding_node(state: Any) -> dict:
    """Generate CLIP embedding — pure image OR text-only."""
    # If there's an image, we use image to search
    if state.get("image_base64"):
        vector = await asyncio.to_thread(embed_image_base64, state["image_base64"])
        return {"image_embedding": vector}
        
    # If no image, we check if there's a text query to search visually
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    # Get the last message content
    last_msg = messages[-1]
    
    # Handle both dict and BaseMessage formats
    if isinstance(last_msg, dict):
        user_text = last_msg.get("content", "")
    else:
        user_text = getattr(last_msg, "content", "")
        
    if not user_text:
        return {}
        
    # Embed the text using CLIP to search for images matching the text
    vector = await asyncio.to_thread(embed_text, user_text)
    return {"image_embedding": vector}
