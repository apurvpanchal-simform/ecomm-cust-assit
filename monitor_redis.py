import asyncio
import redis.asyncio as redis
import os
from dotenv import load_dotenv

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")

async def monitor():
    print(f"Connecting to Redis at {REDIS_URL}...")
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        pubsub = r.pubsub()
        
        # Subscribe to all channels matching the pattern '*'
        await pubsub.psubscribe('*')
        print("✅ Subscribed to all channels! Listening for messages...\n" + "-"*50)
        
        async for message in pubsub.listen():
            if message["type"] == "pmessage":
                print(f"[{message['channel']}] -> {message['data']}")
            else:
                print(f"System event: {message}")
                
    except Exception as e:
        print(f"Error connecting to Redis: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(monitor())
    except KeyboardInterrupt:
        print("\nStopped monitoring.")
