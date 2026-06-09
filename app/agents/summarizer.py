from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from app.graph.state import AgentState
from langsmith import traceable

SUMMARIZER_PROMPT = """You are an ultra-concise conversation summarizer.
Your goal is to aggressively compress older conversation history into a minimal set of facts.

CRITICAL INSTRUCTIONS:
1. Drop all pleasantries, greetings, and conversational filler.
2. Extract ONLY hard facts, user constraints, preferences, and entity IDs.
3. You MUST strictly preserve all specific IDs (like order IDs, customer IDs, and tracking numbers).
4. ALL IDs must be converted to and stored in strictly lowercase (e.g., "ord-123", not "ORD-123").
5. Format your output as a dense, bulleted list of facts. 

If a previous summary is provided, merge the new facts into the existing bulleted list. Do not duplicate facts.
Return ONLY the updated bulleted list.
"""



@traceable(name="summarizer_node")
async def summarizer_node(state: AgentState, config: RunnableConfig) -> dict:
    """Updates the running summary of the conversation if needed."""
    all_messages = state.get("messages", [])
    summarized_count = state.get("summarized_message_count", 0)
    current_summary = state.get("chat_summary", "")
    
    WINDOW_SIZE = 4
    CHUNK_SIZE = 4
    
    # We want to keep the last WINDOW_SIZE messages completely unsummarized.
    # To prevent "telephone" effect, we wait until we have CHUNK_SIZE extra messages
    # before we run the summarizer.
    if len(all_messages) - summarized_count >= WINDOW_SIZE + CHUNK_SIZE:
        # Dynamic chunking: if only a few messages would be left, grab them all
        unsummarized = len(all_messages) - summarized_count
        messages_to_grab = CHUNK_SIZE
        if unsummarized - CHUNK_SIZE < WINDOW_SIZE:
            # Not enough left to form a meaningful window, just grab everything
            # up to the window boundary
            messages_to_grab = unsummarized - WINDOW_SIZE + 1
            
        messages_to_summarize = all_messages[summarized_count : summarized_count + messages_to_grab]
                
        # Format these messages into a readable string
        formatted_messages = []
        for msg in messages_to_summarize:
            msg_type = getattr(msg, "type", "")
            # Skip ToolMessages to avoid polluting the summary with raw tool outputs
            if msg_type == "tool":
                continue
                
            role = "User" if msg_type == "human" else "Assistant"
            content = msg.content
            if content:
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
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"\n========== NEW CHAT SUMMARY ==========\n{new_summary}\n======================================\n")
            
            try:
                import os
                os.makedirs("data", exist_ok=True)
                with open("data/summaries.log", "a", encoding="utf-8") as f:
                    f.write(f"========== SUMMARY ==========\n{new_summary}\n\n")
            except Exception as e:
                logger.warning(f"Could not write to summaries.log: {e}")
                
            # The new summarized count should include all messages we just summarized
            new_summarized_count = summarized_count + len(messages_to_summarize)
                        
            return {
                "chat_summary": new_summary,
                "summarized_message_count": new_summarized_count
            }
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"[SUMMARIZER] LLM Error: {e}")
            return {}
            
    # If no summarization is needed, return an empty dict (state unchanged)
    import logging
    logging.getLogger(__name__).debug(f"[SUMMARIZER] Sleeping. Total msgs: {len(all_messages)}, Summarized: {summarized_count}")
    return {}
