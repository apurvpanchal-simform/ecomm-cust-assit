from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RateLimitConfig(BaseSettings):
    """Configuration for rate limiting.

    Attributes:
        normal_limit: Normal per-minute cap before a 429 is returned.
        spam_threshold: Count at which customer is banned.
        ban_duration_seconds: Duration of the temporary spam ban.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    normal_limit: int = Field(
        default=10, validation_alias="RATE_LIMIT_NORMAL_LIMIT"
    )
    spam_threshold: int = Field(
        default=20, validation_alias="RATE_LIMIT_SPAM_THRESHOLD"
    )
    ban_duration_seconds: int = Field(
        default=86400, validation_alias="RATE_LIMIT_BAN_DURATION_SECONDS"
    )


RATE_LIMIT_CONFIG = RateLimitConfig()
