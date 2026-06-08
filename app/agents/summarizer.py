from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from app.graph.state import AgentState

SUMMARIZER_PROMPT = """You are a helpful assistant responsible for summarizing an ongoing conversation.
Your goal is to maintain a running summary of the conversation so far, capturing the essential context, user requests, and assistant answers.

CRITICAL INSTRUCTIONS:
1. You MUST strictly preserve all specific IDs (like order IDs, customer IDs, and tracking numbers).
2. ALL IDs must be converted to and stored in strictly lowercase (e.g., "ord-123", not "ORD-123").

If a previous summary is provided, you must combine it with the new messages to create a comprehensive updated summary.
Keep the summary concise but ensure no important details (like order numbers, specific questions, or provided answers) are lost.
Return ONLY the updated summary text.
"""

def _transient_cleanup() -> dict:
    """Return a dict that clears bulky transient fields to keep checkpoints lean."""
    return {
        "image_base64": None,
        "image_embedding": None,
        "visual_results": None,
        "image_tags": None,
        "image_azure_url": None,
        "image_is_safe": None,
        "active_filters": None,
        "image_description": None,
        "search_query": None,
    }

async def summarizer_node(state: AgentState, config: RunnableConfig) -> dict:
    """Updates the running summary of the conversation if needed."""
    all_messages = state.get("messages", [])
    summarized_count = state.get("summarized_message_count", 0)
    current_summary = state.get("chat_summary", "")
    
    WINDOW_SIZE = 8
    CHUNK_SIZE = 6
    
    # We want to keep the last WINDOW_SIZE messages completely unsummarized.
    # To prevent "telephone" effect, we wait until we have CHUNK_SIZE extra messages
    # before we run the summarizer.
    if len(all_messages) - summarized_count >= WINDOW_SIZE + CHUNK_SIZE:
        # Get exactly CHUNK_SIZE messages to summarize
        messages_to_summarize = all_messages[summarized_count : summarized_count + CHUNK_SIZE]
                
        # Format these messages into a readable string
        formatted_messages = []
        for msg in messages_to_summarize:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content
            formatted_messages.append(f"{role}: {content}")
            
        new_content_text = "\n\n".join(formatted_messages)
        
        from app.services.llm import get_llm
        llm = get_llm(temperature=0.0)
        
        prompt_messages = [SystemMessage(content=SUMMARIZER_PROMPT)]
        
        if current_summary:
            prompt_messages.append(SystemMessage(content=f"Previous Summary:\n{current_summary}"))
            
        prompt_messages.append(HumanMessage(content=f"New messages to incorporate into the summary:\n{new_content_text}"))
        
        try:
            response = await llm.ainvoke(prompt_messages, config=config)
            new_summary = response.content
            # The new summarized count should include all messages we just summarized
            new_summarized_count = summarized_count + len(messages_to_summarize)
                        
            return {
                "chat_summary": new_summary,
                "summarized_message_count": new_summarized_count,
                # Clear transient fields to keep checkpoints lean
                "image_base64": None,
                "image_embedding": None,
                "visual_results": None,
                "image_tags": None,
                "image_azure_url": None,
                "image_is_safe": None,
                "active_filters": None,
                "image_description": None,
                "search_query": None,
            }
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"[SUMMARIZER] LLM Error: {e}")
            return _transient_cleanup()
            
    # If no summarization is needed, return an empty dict (state unchanged)
    import logging
    logging.getLogger(__name__).debug(f"[SUMMARIZER] Sleeping. Total msgs: {len(all_messages)}, Summarized: {summarized_count}")
    return _transient_cleanup()
