"""
Authentication and basic health routes.
"""

import logging
from fastapi import APIRouter, HTTPException

from app.db.customers import CustomerService
from app.schemas.api import LoginRequest, LoginResponse
from app.services.jwt_auth import generate_jwt

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Simple health check endpoint. Returns {"status": "ok"}."""
    return {"status": "ok"}


@router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate a customer by email and return a JWT access token.

    Args:
        request: Login request body containing the customer's email.

    Returns:
        A JWT access token upon successful authentication.

    Raises:
        HTTPException 401: If no customer exists with the given email.
    """
    customer_service = CustomerService()
    customer = customer_service.get_customer_by_email(request.email)

    if not customer:
        raise HTTPException(status_code=401, detail="Customer not found")

    token = generate_jwt(
        customer_id=customer["customer_id"],
        email=customer["email"],
    )
    return LoginResponse(access_token=token)
