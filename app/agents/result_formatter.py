from typing import Any
from langchain_core.messages import AIMessage

async def result_formatter_node(state: Any) -> dict:
    """Format Qdrant results into a clean assistant message."""
    results = state.get("visual_results", [])

    if not results:
        message = "I couldn't find any similar products in our catalog. Try uploading a clearer photo or describing what you're looking for."
    else:
        lines = ["Here are the most similar products I found:\n"]
        for i, product in enumerate(results, 1):
            lines.append(
                f"{i}. **{product['title']}**\n"
                f"   - Price: ${product['price']}\n"
                f"   - Match Score: {int(product['score'] * 100)}%\n"
                f"   - <img src=\"{product.get('image_url', '')}\" width=\"100\" style=\"border-radius: 8px; margin-top: 5px;\">\n"
            )
        message = "\n".join(lines)

    return {
        "image_base64": None,          # clear raw bytes before checkpointing
        "image_embedding": None,       # clear CLIP vector (768 floats)
        "visual_results": None,        # clear search results payload
        "image_tags": None,
        "image_azure_url": None,
        "image_is_safe": None,
        "messages": [AIMessage(content=message)],
        "summarized_message_count": state.get("summarized_message_count", 0) + 2
    }
