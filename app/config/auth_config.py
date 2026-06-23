import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AuthConfig:
    """Configuration for JWT authentication.

    Attributes:
        secret_key: Secret key used to sign and verify JWTs.
        algorithm: Hashing algorithm used to encode/decode JWTs.
        expiration_hours: Expiration time of generated JWTs in hours.
    """

    secret_key: str | None = os.getenv("JWT_SECRET_KEY")
    algorithm: str = "HS256"
    expiration_hours: int = int(os.getenv("JWT_EXPIRATION_HOURS", "1"))


AUTH_CONFIG = AuthConfig()
