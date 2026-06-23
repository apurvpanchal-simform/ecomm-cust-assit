"""
Pydantic schemas for the conversation summarizer.
"""

from pydantic import BaseModel, Field


class ActiveIssue(BaseModel):
    """Represents a single unresolved customer issue being tracked."""

    issue_description: str = Field(
        description="Description of the active unresolved issue/goal."
    )
    turns_active: int = Field(
        description="Number of consecutive turns the issue has remained unresolved."
    )


class StructuredSummary(BaseModel):
    """
    Structured representation of the conversation summary produced by the LLM.

    The summarizer LLM fills in each field according to the SUMMARIZER_PROMPT rules.
    The output is then converted to a markdown string via `_format_summary_to_markdown`.
    """

    customer_profile_and_preferences: list[str] = Field(
        description="List of user preferences such as sizes, color choices, budget limits, styles."
    )
    mentioned_orders: list[str] = Field(
        description="List of specific order IDs (lowercase), tracking numbers, or refund states mentioned."
    )
    active_issues: list[ActiveIssue] = Field(
        description="List of active unresolved issues/goals."
    )
    resolved_issues: list[str] = Field(
        description="List of issues/questions that have been successfully answered or completed."
    )
    escalate_to_human: bool = Field(
        description=(
            "Set to true ONLY IF the user explicitly asks for a human agent AND there is an "
            "active unresolved issue. Do not escalate if there are no active issues."
        )
    )
