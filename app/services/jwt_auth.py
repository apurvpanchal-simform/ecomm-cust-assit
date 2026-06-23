"""
Utilities for generating and verifying JSON Web Tokens (JWT).
"""

import datetime

import jwt

from app.config import AUTH_CONFIG


def generate_jwt(customer_id: str, email: str) -> str:
    """
    Generates a new JWT for an authenticated customer.

    Args:
        customer_id: The unique customer identifier.
        email: The customer's email address.

    Returns:
        A signed JWT string containing the customer claims and an expiration time.
    """

    payload = {
        "sub": customer_id,
        "email": email,
        "exp": (
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(hours=AUTH_CONFIG.expiration_hours)
        ),
    }

    return jwt.encode(
        payload,
        AUTH_CONFIG.secret_key,
        algorithm=AUTH_CONFIG.algorithm,
    )


def verify_jwt(token: str) -> dict:
    """
    Verifies and decodes a JWT token.

    Args:
        token: The signed JWT string.

    Returns:
        The decoded payload dictionary.

    Raises:
        jwt.PyJWTError: If the token is invalid, expired, or corrupted.
    """

    return jwt.decode(
        token,
        AUTH_CONFIG.secret_key,
        algorithms=[AUTH_CONFIG.algorithm],
    )
