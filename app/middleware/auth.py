"""
Authentication middleware for verifying JWT tokens and extracting customer IDs.
"""

from fastapi import Depends, HTTPException
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from app.services.jwt_auth import (
    verify_jwt,
)

security = HTTPBearer()


def get_current_customer(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    FastAPI dependency to verify a JWT token and extract the customer ID.

    Args:
        credentials: The Bearer token extracted from the Authorization header.

    Returns:
        The customer ID extracted from the token.

    Raises:
        HTTPException: If the token is invalid or expired.
    """

    try:
        payload = verify_jwt(credentials.credentials)

        return payload["sub"]

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
        )
