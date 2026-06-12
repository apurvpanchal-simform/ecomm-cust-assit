from typing import Any
from langfuse import observe


@observe(name="cleanup_node")
async def cleanup_node(state: Any) -> dict:
    """Lightweight node that clears heavy transient fields to keep checkpoints lean.
    Runs unconditionally after the visual search pipeline finishes.
    """
    return {
        "image_base64": None,
        "image_embedding": None,
        "image_results": None,
        "image_tags": None,
        "image_azure_url": None,
        "image_is_safe": None,
        "active_filters": None,
        "image_description": None,
        "search_query": None,
        "sub_queries": None,
    }
