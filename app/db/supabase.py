"""
Supabase client initialization and management.
"""

import os

from dotenv import load_dotenv
from supabase import AsyncClient, create_async_client

load_dotenv()

_supabase_async: AsyncClient = None


async def get_supabase_client() -> AsyncClient:
    """
    Returns a singleton instance of the asynchronous Supabase client.

    Returns:
        The configured Supabase AsyncClient.
    """
    global _supabase_async
    if _supabase_async is None:
        _supabase_async = await create_async_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY"),
        )
    return _supabase_async
