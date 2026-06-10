from typing import Optional, List
from pydantic import BaseModel, Field

class SearchConstraints(BaseModel):
    search_query: str = Field(description="A clean, optimized search phrase for text-based product matching.")
    min_price: Optional[float] = Field(default=None, description="Minimum price if specified.")
    max_price: Optional[float] = Field(default=None, description="Maximum price if specified.")

class RerankedResult(BaseModel):
    product_id: str = Field(description="The ID of the product")
    relevance_score: float = Field(description="Score from 0.0 to 1.0 on how well it matches the user constraints")
    reasoning: str = Field(description="Why it matches or fails")

class RerankResponse(BaseModel):
    results: List[RerankedResult]
