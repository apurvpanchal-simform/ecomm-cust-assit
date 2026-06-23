"""
Main FastAPI application entry point for the E-Commerce Customer Assistant.

Responsibilities:
- FastAPI application initialization.
- Middleware configuration (CORS).
- Router registration.
- Lifespan management delegation.
"""

import logging
from dotenv import load_dotenv

# Load environment variables first before importing modules
load_dotenv(override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.lifespan import lifespan
from app.routers import auth, chat_ws, conversations, data, support

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router)
app.include_router(conversations.router)
app.include_router(data.router)
app.include_router(chat_ws.router)
app.include_router(support.router)
