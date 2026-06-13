"""
Pydantic schemas for structured LLM routing and visual search reranking.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SearchConstraints(BaseModel):
    """Schema for defining constraints when routing to the image search agent."""

    search_query: str = Field(
        description="A clean, optimized search phrase for text-based product matching."
    )
    min_price: Optional[float] = Field(
        default=None, description="Minimum price if specified."
    )
    max_price: Optional[float] = Field(
        default=None, description="Maximum price if specified."
    )


class RerankedResult(BaseModel):
    """Schema for a single product evaluated and reranked by the LLM."""

    product_id: str = Field(description="The ID of the product")
    relevance_score: float = Field(
        description="Score from 0.0 to 1.0 on how well it matches the user constraints"
    )
    reasoning: str = Field(description="Why it matches or fails")


class RerankResponse(BaseModel):
    """Schema for the final reranked list of product matches."""

    results: List[RerankedResult]
