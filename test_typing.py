import asyncio
import json
import websockets
import jwt
from datetime import datetime, timedelta, timezone
import httpx

secret = "super-secret-key-that-is-at-least-32-chars-long"
payload = {
    "sub": "cust-001",
    "email": "test@gmail.com",
    "exp": datetime.now(tz=timezone.utc) + timedelta(days=15)
}
token = jwt.encode(payload, secret, algorithm="HS256")

async def test():
    uri = f"ws://localhost:8000/chat/ws?token={token}&conversation_id=test-conv-1"
    try:
        async with websockets.connect(uri) as ws:
            print("Connected to WS!")
            
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    "http://localhost:8000/support/conversations/test-conv-1/typing",
                    json={"agent_name": "Agent Alex"}
                )
                print("Typing POST:", res.status_code, res.text)
            
            # Read from websocket
            while True:
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    print("WS Received:", data)
                except asyncio.TimeoutError:
                    print("Finished waiting.")
                    break
    except Exception as e:
        print("Error:", e)

asyncio.run(test())
