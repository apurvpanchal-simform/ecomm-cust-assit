from typing import Optional
from pydantic import BaseModel, Field
from langsmith import traceable
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from app.graph.state import AgentState
from app.services.llm import get_llm
import logging

logger = logging.getLogger(__name__)

FILTER_EXTRACTOR_PROMPT = """You are an e-commerce search assistant.
Your job is to read the user's conversation, analyze what they are looking for (including any uploaded images), and extract a clean search query and filters.

Rules:
1. Extract a clear `search_query` that combines the user's intent, the image description (if available), and relevant conversation context.
2. If the user mentions a maximum budget (e.g. "under $50", "cheaper than 100"), set `max_price`.
3. If the user mentions a minimum budget (e.g. "over $20"), set `min_price`.
4. If they just say "find this" and an image is present, the search query should heavily rely on the image description.
5. Do NOT include price terms in the `search_query` itself (e.g. "red shoes" instead of "red shoes under 50").
"""

class SearchConstraints(BaseModel):
    search_query: str = Field(description="A clean, optimized search phrase for text-based product matching.")
    min_price: Optional[float] = Field(default=None, description="Minimum price if specified.")
    max_price: Optional[float] = Field(default=None, description="Maximum price if specified.")

@traceable(name="filter_extractor_node")
async def filter_extractor_node(state: AgentState, config: RunnableConfig) -> dict:
    """Extracts constraints and builds a smart search query before visual search."""
    
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(SearchConstraints)
    
    messages_to_send = [SystemMessage(content=FILTER_EXTRACTOR_PROMPT)]
    
    chat_summary = state.get("chat_summary", "")
    if chat_summary:
        messages_to_send.append(SystemMessage(content=f"Summary of earlier conversation:\n{chat_summary}"))
        
    image_desc = state.get("image_description")
    if image_desc:
        messages_to_send.append(SystemMessage(content=f"The user uploaded an image. Image Analysis:\n{image_desc}"))
        
    # Send all unsummarized messages to ensure context
    all_messages = state.get("messages", [])
    summarized_count = state.get("summarized_message_count", 0)
    recent_messages = all_messages[summarized_count:]
    
    for msg in recent_messages:
        messages_to_send.append(msg)
        
    try:
        response = await structured_llm.ainvoke(messages_to_send, config=config)
        
        active_filters = {}
        if response.min_price is not None:
            active_filters["min_price"] = response.min_price
        if response.max_price is not None:
            active_filters["max_price"] = response.max_price
            
        return {
            "search_query": response.search_query,
            "active_filters": active_filters if active_filters else None
        }
    except Exception as e:
        logger.exception(f"Filter extractor failed: {e}")
        # Fallback if LLM fails
        return {
            "search_query": getattr(recent_messages[-1], "content", "") if recent_messages else "",
            "active_filters": None
        }
