from typing import Any

def result_formatter_node(state: Any) -> dict:
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
                f"   - ![Product Image]({product.get('image_url', '')})\n"
            )
        message = "\n".join(lines)

    return {
        "image_base64": None,          # clear raw bytes before checkpointing
        "messages": [{"role": "assistant", "content": message}]
    }
