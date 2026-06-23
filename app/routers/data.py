"""
Data pass-through routes for fetching Supabase records.
"""

from fastapi import APIRouter, Depends

from app.db.supabase import get_supabase_client
from app.middleware.rate_limit import rate_limit_customer

router = APIRouter()


@router.get("/orders")
async def get_orders(customer_id: str = Depends(rate_limit_customer)):
    """Fetch raw order rows for the authenticated customer from Supabase."""
    supabase = await get_supabase_client()
    result = (
        await supabase.table("orders")
        .select("*")
        .eq("customer_id", customer_id)
        .order("ordered_at", desc=True)
        .execute()
    )
    return result.data


@router.get("/products")
async def get_products(customer_id: str = Depends(rate_limit_customer)):
    """Fetch all products from Supabase."""
    supabase = await get_supabase_client()
    result = await supabase.table("products").select("*").execute()
    return result.data
