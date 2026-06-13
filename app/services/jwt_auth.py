"""
Utilities for generating and verifying JSON Web Tokens (JWT).
"""

import datetime
import os

import jwt

JWT_SECRET = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"


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
            + datetime.timedelta(hours=int(os.getenv("JWT_EXPIRATION_HOURS", "1")))
        ),
    }

    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
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
        JWT_SECRET,
        algorithms=[JWT_ALGORITHM],
    )
