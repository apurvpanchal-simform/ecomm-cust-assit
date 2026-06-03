from fastapi import FastAPI
from pydantic import BaseModel
from app.graph.builder import graph
from fastapi import HTTPException
from app.schemas.auth import (LoginRequest,LoginResponse,)
from app.schemas.input import (ChatRequest,)
from app.services.auth_service import (generate_jwt,)
from app.services.customer_service import (CustomerService,)
from fastapi import Depends
from app.middleware.auth import (get_current_customer)

app = FastAPI()

@app.post(
    "/auth/login",
    response_model=LoginResponse
)
async def login(
    request: LoginRequest
):

    customer_service = CustomerService()

    customer = (
        customer_service.get_customer_by_email(
            request.email
        )
    )

    if not customer:
        raise HTTPException(
            status_code=401,
            detail="Customer not found",
        )

    token = generate_jwt(
        customer_id=customer["customer_id"],
        email=customer["email"],
    )

    return LoginResponse(
        access_token=token
    )

@app.post("/chat")
async def chat(
    request: ChatRequest,
    customer_id: str = Depends(
        get_current_customer
    ),
):

    result = graph.invoke(
        {
            "query": request.query,
            "customer_id": customer_id,
        }
    )

    return result