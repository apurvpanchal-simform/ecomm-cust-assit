from fastapi import Depends, Request, HTTPException
from app.middleware.auth import get_current_customer

async def rate_limit_customer(
    request: Request,
    customer_id: str = Depends(get_current_customer)
) -> str:
    """Rate limit to 10 requests per minute per customer."""
    redis_client = getattr(request.app.state, "redis", None)
    if not redis_client:
        return customer_id
        
    key = f"rate_limit:{customer_id}"
    
    current = await redis_client.get(key)
    if current and int(current) >= 10:
        raise HTTPException(
            status_code=429, 
            detail="Too Many Requests. Please wait a minute before trying again."
        )
        
    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.incr(key)
        pipe.expire(key, 60, nx=True)
        await pipe.execute()
        
    return customer_id
