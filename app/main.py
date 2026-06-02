from fastapi import FastAPI
from pydantic import BaseModel
from app.graph.builder import graph

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    result = graph.invoke({"query": request.message})
    return result