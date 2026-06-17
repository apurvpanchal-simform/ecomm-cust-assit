import asyncio
from dotenv import load_dotenv
load_dotenv()

from app.services.faq_response_cache import get_faq_response

async def main():
    res = await get_faq_response("test_cust", "what is your return policy?")
    print("Result:", res)

asyncio.run(main())
