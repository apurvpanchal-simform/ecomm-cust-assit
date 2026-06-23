import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for hybrid and multimodal search parameters.

    Attributes:
        prefetch_limit: The maximum candidate points retrieved per modality prefetch query.
        image_score_threshold: Minimum match score required for dense image vector matches.
        text_score_threshold: Minimum match score required for dense text vector matches.
        log_limit: Limit of records to fetch when printing individual vector debug results.
        fusion_limit: Maximum final candidate points to return after RRF fusion.
        faq_search_threshold: Minimum confidence score required for FAQ retrieval.
        faq_cache_threshold: Similarity threshold required for semantic FAQ cache hit.
        faq_top_k: Number of FAQ chunks to retrieve in vector search.
    """

    prefetch_limit: int = 20
    image_score_threshold: float = 0.18
    text_score_threshold: float = 0.22
    log_limit: int = 5
    fusion_limit: int = 3
    faq_search_threshold: float = 0.50
    faq_cache_threshold: float = field(
        default_factory=lambda: float(os.getenv("FAQ_SEMANTIC_CACHE_THRESHOLD", "0.75"))
    )
    faq_top_k: int = 5


SEARCH_CONFIG = SearchConfig()
