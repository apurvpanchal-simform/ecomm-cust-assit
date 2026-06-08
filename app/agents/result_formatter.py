from typing import Any
from langchain_core.messages import AIMessage, SystemMessage
from app.services.llm import get_llm
import json

FORMATTER_PROMPT = """You are a helpful e-commerce assistant.
The system just performed a visual search or product search based on the user's query and/or uploaded image.
Below are the results found in the catalog.

Present these results to the user in a friendly, conversational manner.
Answer any specific questions the user had (e.g., "Which is cheapest?").
If no results were found, politely apologize and suggest trying a different query or a clearer image.

Formatting rules for products:
- Always use Markdown.
- Include the price.
- Display the image using an HTML img tag: <img src="URL" width="150" style="border-radius: 8px; margin-top: 5px;">
"""

async def result_formatter_node(state: Any) -> dict:
    """Format Qdrant results using an LLM to be conversational."""
    results = state.get("visual_results", [])
    
    llm = get_llm(temperature=0.4)
    
    messages_to_send = [SystemMessage(content=FORMATTER_PROMPT)]
    
    chat_summary = state.get("chat_summary", "")
    if chat_summary:
        messages_to_send.append(SystemMessage(content=f"Summary of earlier conversation:\n{chat_summary}"))
        
    image_desc = state.get("image_description")
    if image_desc:
        messages_to_send.append(SystemMessage(content=f"The user uploaded an image. Image Analysis:\n{image_desc}"))
        
    # Add the JSON search results as system context
    messages_to_send.append(SystemMessage(content=f"Search Results from Catalog:\n{json.dumps(results, indent=2)}"))
    
    # Send all unsummarized messages to ensure context
    all_messages = list(state.get("messages", []))
    summarized_count = state.get("summarized_message_count", 0)
    recent_messages = all_messages[summarized_count:]
    
    for msg in recent_messages:
        messages_to_send.append(msg)
        
    try:
        response = await llm.ainvoke(messages_to_send)
        message = response.content
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception(f"Result formatter LLM failed: {e}")
        message = "I found some products, but encountered an error formatting them. Please try again."

    # If the last message was an AI message, we merge this response (e.g. if we are resuming)
    last_msg = all_messages[-1] if all_messages else None
    if last_msg and hasattr(last_msg, 'type') and last_msg.type == "ai":
        merged_content = f"{last_msg.content}\n\n{message}"
        new_messages = [AIMessage(content=merged_content, id=last_msg.id)]
    else:
        new_messages = [AIMessage(content=message)]

    return {
        "image_base64": None,          # clear raw bytes before checkpointing
        "image_embedding": None,       # clear CLIP vector (768 floats)
        "visual_results": None,        # clear search results payload
        "image_tags": None,
        "image_azure_url": None,
        "image_is_safe": None,
        "messages": new_messages,
        "executed_agents": state.get("executed_agents", []) + ["visual_search_agent"]
    }
