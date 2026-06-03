import os
import jwt
import datetime


JWT_SECRET = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"


def generate_jwt(
    customer_id: str,
    email: str
) -> str:

    payload = {
        "sub": customer_id,
        "email": email,
        "exp": (
            datetime.datetime.now(
                datetime.timezone.utc
            )
            + datetime.timedelta(
                hours=int(
                    os.getenv(
                        "JWT_EXPIRATION_HOURS",
                        "1"
                    )
                )
            )
        ),
    }

    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def verify_jwt(
    token: str
) -> dict:

    return jwt.decode(
        token,
        JWT_SECRET,
        algorithms=[JWT_ALGORITHM],
    )