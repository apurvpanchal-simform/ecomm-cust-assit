"""Re-exports all agent system prompts from the single source of truth: prompts.py."""

from app.prompts.prompts import (
    FAQ_SYSTEM_PROMPT,
    ORDER_SYSTEM_PROMPT,
    SUMMARIZER_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    SYNTHESIZER_PROMPT,
    get_supervisor_system_prompt,
    get_faq_system_prompt,
    get_order_system_prompt,
)

__all__ = [
    "SUPERVISOR_SYSTEM_PROMPT",
    "FAQ_SYSTEM_PROMPT",
    "ORDER_SYSTEM_PROMPT",
    "SUMMARIZER_PROMPT",
    "SYNTHESIZER_PROMPT",
    "get_supervisor_system_prompt",
    "get_faq_system_prompt",
    "get_order_system_prompt",
]
