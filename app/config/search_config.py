from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SearchConfig(BaseSettings):
    """Configuration for hybrid and multimodal search parameters.

    Every threshold can be overridden via an environment variable so that
    values can be tuned in production without a code change or redeploy.

    Attributes:
        prefetch_limit: Max candidate points retrieved per modality prefetch query.
        image_score_threshold: Minimum cosine similarity for dense image prefetch
            (SigLIP2 cross-modal scores typically range 0.01–0.20).
        text_score_threshold: Minimum cosine similarity for dense text prefetch
            (SigLIP2 text-only scores typically range 0.01–0.10).
        rrf_score_threshold: Minimum RRF fusion score to keep a result.
            Candidates below this are discarded.
        log_limit: Number of records to log per individual vector debug query.
        fusion_limit: Max final candidates to return after RRF fusion.
        faq_search_threshold: Minimum confidence for FAQ retrieval.
        faq_cache_threshold: Similarity threshold for semantic FAQ cache hit.
        faq_top_k: Number of FAQ chunks to retrieve in vector search.
        clip_model: SigLIP model name.
        hf_token: HuggingFace user access token.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    prefetch_limit: int = Field(
        default=10, validation_alias="SEARCH_PREFETCH_LIMIT"
    )
    image_score_threshold: float = Field(
        default=0.02, validation_alias="SEARCH_IMAGE_SCORE_THRESHOLD"
    )
    text_score_threshold: float = Field(
        default=0.02, validation_alias="SEARCH_TEXT_SCORE_THRESHOLD"
    )
    rrf_score_threshold: float = Field(
        default=0.15, validation_alias="SEARCH_RRF_SCORE_THRESHOLD"
    )
    log_limit: int = Field(
        default=5, validation_alias="SEARCH_LOG_LIMIT"
    )
    fusion_limit: int = Field(
        default=3, validation_alias="SEARCH_FUSION_LIMIT"
    )
    faq_search_threshold: float = Field(
        default=0.50, validation_alias="FAQ_SEARCH_THRESHOLD"
    )
    faq_cache_threshold: float = Field(
        default=0.75, validation_alias="FAQ_SEMANTIC_CACHE_THRESHOLD"
    )
    faq_top_k: int = Field(
        default=5, validation_alias="FAQ_TOP_K"
    )
    clip_model: str = Field(
        default="google/siglip-base-patch16-224", validation_alias="CLIP_MODEL"
    )
    hf_token: str | None = Field(
        default=None, validation_alias="HF_TOKEN"
    )


SEARCH_CONFIG = SearchConfig()
