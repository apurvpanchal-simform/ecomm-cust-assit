from fastapi import Depends
from fastapi import HTTPException

from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
)

from app.services.jwt_auth import (
    verify_jwt,
)

security = HTTPBearer()


def get_current_customer(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:

    try:

        payload = verify_jwt(credentials.credentials)

        return payload["sub"]

    except Exception:

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
        )
