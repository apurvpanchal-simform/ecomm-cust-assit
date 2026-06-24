from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthConfig(BaseSettings):
    """Configuration for JWT authentication.

    Attributes:
        secret_key: Secret key used to sign and verify JWTs.
        algorithm: Hashing algorithm used to encode/decode JWTs.
        expiration_hours: Expiration time of generated JWTs in hours.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    secret_key: str | None = Field(
        default=None, validation_alias="JWT_SECRET_KEY"
    )
    algorithm: str = Field(
        default="HS256", validation_alias="JWT_ALGORITHM"
    )
    expiration_hours: int = Field(
        default=1, validation_alias="JWT_EXPIRATION_HOURS"
    )


AUTH_CONFIG = AuthConfig()
