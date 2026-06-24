"""
Prompts package — public API.

One file per agent. Import all prompt builder functions from here.
To tune any prompt or framing message, edit the corresponding agent prompt file.
"""

from app.prompts.supervisor import (
    get_supervisor_system_prompt,
    get_chat_summary_message as get_supervisor_chat_summary_message,
    get_image_context_message,
    get_image_safety_warning_message,
    get_latest_conversation_header,
)
from app.prompts.faq import (
    get_faq_system_prompt,
    get_chat_summary_message as get_faq_chat_summary_message,
    get_sub_query_message as get_faq_sub_query_message,
)
from app.prompts.order import (
    get_order_system_prompt,
    get_chat_summary_message as get_order_chat_summary_message,
    get_sub_query_message as get_order_sub_query_message,
)
from app.prompts.summarizer import (
    get_summarizer_prompt,
    get_previous_summary_message,
    get_new_messages_input,
)
from app.prompts.synthesizer import (
    get_synthesizer_prompt,
    get_synthesizer_system_message,
    get_agent_responses_message,
)

__all__ = [
    # Supervisor
    "get_supervisor_system_prompt",
    "get_supervisor_chat_summary_message",
    "get_image_context_message",
    "get_image_safety_warning_message",
    "get_latest_conversation_header",
    # FAQ
    "get_faq_system_prompt",
    "get_faq_chat_summary_message",
    "get_faq_sub_query_message",
    # Order
    "get_order_system_prompt",
    "get_order_chat_summary_message",
    "get_order_sub_query_message",
    # Summarizer
    "get_summarizer_prompt",
    "get_previous_summary_message",
    "get_new_messages_input",
    # Synthesizer
    "get_synthesizer_prompt",
    "get_synthesizer_system_message",
    "get_agent_responses_message",
]

