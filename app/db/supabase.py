from supabase import create_async_client, AsyncClient
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

_supabase_async: AsyncClient = None


async def get_supabase_client() -> AsyncClient:
    global _supabase_async
    if _supabase_async is None:
        _supabase_async = await create_async_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY"),
        )
    return _supabase_async
