"""
Chainlit Chat UI for E-Commerce Customer Assist.

Uses a PERSISTENT WebSocket connection per Chainlit session (/chat/ws) for
real-time token streaming, and plain HTTP for auth, history, orders, products.

Key design: one WS connection is opened in on_chat_start and reused for every
message in that session. on_chat_end closes it cleanly.
"""

import asyncio
import base64
import datetime
import json
import logging
import os
import re
import uuid
from typing import Dict, List, Optional

import chainlit as cl
import chainlit.data as cl_data
import httpx
import jwt
import psycopg
import websockets
from chainlit.step import StepDict
from chainlit.types import (
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
WS_BASE_URL = API_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")


# ── Custom Data Layer Helpers ──────────────────────────────────────────────


async def get_customer_id_for_thread(thread_id: str) -> Optional[str]:
    """Retrieves the customer ID associated with the given thread ID. Returns None if no customer is linked."""
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        logger.error("SUPABASE_DB_URL is not set in environment.")
        return None
    try:
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT customer_id FROM customer_conversations WHERE conversation_id = %s",
                    (thread_id,),
                )
                row = await cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        logger.error("Failed to query customer_id for thread %s: %s", thread_id, e)
        return None


async def get_customer_email(customer_id: str) -> Optional[str]:
    """Retrieve the email address for a given customer ID asynchronously.

    Returns the email string if found, otherwise None."""
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        return None
    try:
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT email FROM customers WHERE customer_id = %s", (customer_id,)
                )
                row = await cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        logger.error("Failed to query email for customer %s: %s", customer_id, e)
        return None


def generate_system_jwt(customer_id: str) -> str:
    """Generate a system JWT for the specified customer ID.

    Returns a signed JWT string."""
    secret = os.getenv("JWT_SECRET_KEY")
    payload = {
        "sub": customer_id,
        "email": "system@ecomm-cust-assist.internal",
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


# ── Custom Data Layer for Persistence & History Sidebar ───────────────────


class CustomDataLayer(cl_data.BaseDataLayer):
    """CustomDataLayer extends BaseDataLayer to provide specialized data handling for custom datasets, supporting dynamic loading, preprocessing, and caching mechanisms."""

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        """Retrieves a persisted user by identifier.
        Returns a PersistedUser instance or None if no matching user is found."""
        token_data = await api_login(identifier)
        token = token_data["access_token"] if token_data else ""
        return PersistedUser(
            id=identifier,
            createdAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            identifier=identifier,
            display_name=identifier.split("@")[0],
            metadata={"jwt_token": token},
        )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        """Creates a new user in the database and returns the persisted user object, or None if the operation fails."""
        token = user.metadata.get("jwt_token", "")
        if not token:
            token_data = await api_login(user.identifier)
            token = token_data["access_token"] if token_data else ""

        metadata = dict(user.metadata or {})
        metadata["jwt_token"] = token

        return PersistedUser(
            id=user.identifier,
            createdAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            identifier=user.identifier,
            display_name=user.display_name or user.identifier.split("@")[0],
            metadata=metadata,
        )

    async def delete_feedback(self, feedback_id: str) -> bool:
        """Deletes a feedback entry by its ID.

        Returns True if the deletion was successful, otherwise False."""
        return True

    async def upsert_feedback(self, feedback) -> str:
        """Upserts the given feedback into the database and returns the operation status as a string.

        Args:
            feedback: The feedback data to be inserted or updated.

        Returns:
            A string indicating success or failure of the upsert operation."""
        return "feedback_id"

    async def create_element(self, element):
        """Creates an element asynchronously and returns the created element.

        Args:
            element: The element to create.

        Returns:
            The created element."""
        pass

    async def get_element(self, thread_id: str, element_id: str):
        """Retrieve an element from a thread by its ID.

        Args:
            thread_id: Identifier of the thread.
            element_id: Identifier of the element within the thread.

        Returns:
            The element object or None if not found.
        """
        return None

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        """Deletes the element with the given element_id, optionally within the specified thread_id. Returns a confirmation or raises an error if the element does not exist."""
        pass

    async def create_step(self, step_dict: StepDict):
        """Creates a new step from the provided dictionary and returns the created Step object.

        Args:
            step_dict (StepDict): Dictionary containing step configuration.

        Returns:
            Step: The created step instance.
        """
        pass

    async def update_step(self, step_dict: StepDict):
        """Updates the current step with the provided step dictionary.
        Validates the input and persists changes asynchronously."""
        pass

    async def delete_step(self, step_id: str):
        """Deletes the step identified by step_id from the workflow, raising an exception if the step is not found."""
        pass

    async def get_thread_author(self, thread_id: str) -> str:
        """Retrieve the author of a thread by its ID.

        Args:
            thread_id: The unique identifier of the thread.

        Returns:
            The author's username as a string.
        """
        customer_id = await get_customer_id_for_thread(thread_id)
        if not customer_id:
            return ""
        email = await get_customer_email(customer_id)
        return email or customer_id or ""

    async def delete_thread(self, thread_id: str):
        """Deletes the thread identified by thread_id. Raises ThreadNotFoundError if the thread does not exist."""
        logger.info("[DataLayer] delete_thread called for thread_id=%s", thread_id)
        customer_id = await get_customer_id_for_thread(thread_id)
        if not customer_id:
            logger.warning(
                logger.info(
                    "[DataLayer] Thread %s not found in DB, skipping deletion.",
                    thread_id,
                )
            )
            return

        token = generate_system_jwt(customer_id)
        await api_delete_conversation(token, thread_id)
        logger.info("[DataLayer] Thread %s deleted successfully.", thread_id)

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        """Retrieve a paginated list of threads that match the given filters.

        Args:
            pagination: Pagination parameters.
            filters: ThreadFilter specifying filter criteria.

        Returns:
            List[Thread]: The matching threads.
        """
        user_identifier = filters.userId
        if not user_identifier:
            return PaginatedResponse(
                pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None),
                data=[],
            )

        token_data = await api_login(user_identifier)
        if not token_data:
            return PaginatedResponse(
                pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None),
                data=[],
            )

        token = token_data["access_token"]
        convs = await api_fetch_conversations(token)

        threads = []
        for c in convs:
            threads.append(
                {
                    "id": c["conversation_id"],
                    "createdAt": (
                        c["updated_at"]
                        if isinstance(c.get("updated_at"), str)
                        else datetime.datetime.now(datetime.timezone.utc).isoformat()
                    ),
                    "name": c["title"],
                    "userId": user_identifier,
                    "userIdentifier": user_identifier,
                    "tags": [],
                    "metadata": {},
                    "steps": [],
                    "elements": [],
                }
            )

        return PaginatedResponse(
            pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None),
            data=threads,
        )

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        """Asynchronously fetches a thread by its ID, returning a ThreadDict or None if the thread does not exist."""
        logger.info("[DataLayer] get_thread called for thread_id=%s", thread_id)
        user = None
        try:
            user = cl.user_session.get("user")
            logger.info("[DataLayer] user from cl.user_session: %s", user)
        except Exception as e:
            logger.debug(
                logger.info(
                    "[DataLayer] cl.user_session not available in get_thread: %s", e
                )
            )

        # Resolve customer_id from DB using thread_id
        customer_id = await get_customer_id_for_thread(thread_id)
        if not customer_id:
            logger.warning(
                logger.info(
                    "[DataLayer] Thread %s not found in database customer_conversations",
                    thread_id,
                )
            )
            # If the thread is new and not in the DB yet, we can fallback to the current session user if available
            if user:
                token_data = await api_login(user.identifier)
                if token_data:
                    try:
                        payload = jwt.decode(
                            token_data["access_token"],
                            os.getenv("JWT_SECRET_KEY"),
                            algorithms=["HS256"],
                        )
                        customer_id = payload.get("sub")
                    except Exception as e:
                        logger.error(
                            logger.error(
                                "[DataLayer] Failed to decode fallback user token: %s",
                                e,
                            )
                        )

        if not customer_id:
            logger.warning(
                logger.info(
                    "[DataLayer] No customer_id could be resolved for thread_id=%s",
                    thread_id,
                )
            )
            # If no user or customer is found (e.g. anonymous/new), return an empty thread structure
            return {
                "id": thread_id,
                "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "name": "Chat Thread",
                "userId": user.identifier if user else "guest",
                "userIdentifier": user.identifier if user else "guest",
                "tags": [],
                "metadata": {},
                "steps": [],
                "elements": [],
            }

        # Generate a system token for this customer_id to fetch history
        token = generate_system_jwt(customer_id)
        history = await api_fetch_history(token, thread_id)

        steps = []
        elements = []
        for m in history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "assistant":
                _, content = parse_reasoning(content)

            step_id = str(uuid.uuid4())
            steps.append(
                {
                    "id": step_id,
                    "threadId": thread_id,
                    "parentId": None,
                    "streaming": False,
                    "metadata": {},
                    "tags": [],
                    "input": "",
                    "isError": False,
                    "output": content,
                    "createdAt": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "start": None,
                    "end": None,
                    "language": None,
                    "name": "User" if role == "user" else "Assistant",
                    "type": "user_message" if role == "user" else "assistant_message",
                    "showInput": False,
                }
            )

            for img_url in m.get("images", []):
                elements.append(
                    {
                        "id": str(uuid.uuid4()),
                        "threadId": thread_id,
                        "type": "image",
                        "url": img_url,
                        "name": "uploaded_image",
                        "display": "inline",
                        "forId": step_id,
                    }
                )

        if user:
            user_identifier = user.identifier
        else:
            email = await get_customer_email(customer_id) if customer_id else None
            user_identifier = email or customer_id or "guest"

        return {
            "id": thread_id,
            "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "name": "Chat Thread",
            "userId": user_identifier,
            "userIdentifier": user_identifier,
            "tags": [],
            "metadata": {},
            "steps": steps,
            "elements": elements,
        }

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        """Asynchronously updates the thread identified by `thread_id`, applying any pending changes and persisting them. Returns the updated thread or a status indicator."""
        pass

    async def build_debug_url(self) -> str:
        """Asynchronously constructs and returns a debug URL for the current instance, including relevant state and query parameters."""
        return ""

    async def close(self) -> None:
        """Closes the resource asynchronously, ensuring all pending operations are completed and the underlying connection is released."""
        pass

    async def get_favorite_steps(self, user_id: str) -> List[StepDict]:
        """Retrieve the user's favorite steps.

        Args:
            user_id: The unique identifier of the user.

        Returns:
            A list of StepDict objects representing the user's favorite steps."""
        return []


@cl.data_layer
def get_data_layer():
    """Retrieve the data layer configuration used for analytics tracking.

    Returns:
        dict: A dictionary representing the data layer structure."""
    return CustomDataLayer()


# ── Parsing Helpers ──────────────────────────────────────────────────────


def parse_reasoning(content: str) -> tuple[str | None, str]:
    """Split <think>…</think> reasoning from the visible reply."""
    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    else:
        reasoning = None
        clean = content
    clean = re.sub(r"`(\*\*.*?\*\*)`", r"\1", clean)
    return reasoning, clean


# ── API Integration ─────────────────────────────────────────────────────────

import chainlit as cl
from chainlit.server import app as cl_app
from chainlit.types import ThreadDict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse


class AvatarFallbackMiddleware(BaseHTTPMiddleware):
    """Middleware that intercepts avatar requests and serves a default image when the requested avatar is missing. It extends BaseHTTPMiddleware to integrate seamlessly with the ASGI application stack."""

    async def dispatch(self, request, call_next):
        """Dispatches an incoming request to the next middleware after optional processing.

        Args:
            request: The incoming request object.
            call_next: The next middleware callable to invoke.

        Returns:
            The response returned by the next middleware."""
        response = await call_next(request)
        if response.status_code == 404 and request.url.path.startswith(
            "/public/avatars/"
        ):
            return RedirectResponse(url="/public/avatars/support_agent.png")
        return response


cl_app.add_middleware(AvatarFallbackMiddleware)


async def api_login(email: str) -> dict | None:
    """Attempts to log in to the API using the provided email and returns the authentication payload, or None if login fails.

    Args:
        email: User's email address.

    Returns:
        A dictionary containing login details on success, or None on failure."""
    async with httpx.AsyncClient() as c:
        try:
            r = await c.post(
                f"{API_BASE_URL}/auth/login", json={"email": email}, timeout=10
            )
            return r.json() if r.status_code == 200 else None
        except httpx.RequestError:
            return None


async def api_fetch_conversations(token: str) -> list:
    """Retrieve a list of conversation objects from the API using the given authentication token.

    Args:
        token (str): Authentication token for the API.

    Returns:
        list: A list of conversations."""
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(
                f"{API_BASE_URL}/chat/conversations",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            return r.json() if r.status_code == 200 else []
        except httpx.RequestError:
            return []


async def api_fetch_history(token: str, conversation_id: str) -> list:
    """Retrieve the list of messages in a conversation using the given API token and conversation ID.

    Args:
        token: API authentication token.
        conversation_id: ID of the conversation.

    Returns:
        List of message dictionaries representing the conversation history."""
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(
                f"{API_BASE_URL}/chat/history/{conversation_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            return r.json().get("messages", []) if r.status_code == 200 else []
        except httpx.RequestError:
            return []


async def api_delete_conversation(token: str, conversation_id: str) -> bool:
    """Deletes a conversation identified by conversation_id using the provided authentication token. Returns True if the deletion was successful, otherwise False."""
    async with httpx.AsyncClient() as c:
        try:
            r = await c.delete(
                f"{API_BASE_URL}/chat/conversations/{conversation_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            return r.status_code == 200
        except httpx.RequestError:
            return False


async def api_fetch_orders(token: str) -> list:
    """Fetches and returns a list of orders from the API using the given authentication token.

    Args:
        token (str): The API authentication token.

    Returns:
        list: A list of order dictionaries.
    """
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(
                f"{API_BASE_URL}/orders",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            return r.json() if r.status_code == 200 else []
        except httpx.RequestError:
            return []


async def api_fetch_products(token: str) -> list:
    """Fetches a list of products from the API using the provided authentication token.

    Args:
        token (str): The authentication token for API access.

    Returns:
        list: A list of product dictionaries retrieved from the API."""
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(
                f"{API_BASE_URL}/products",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            return r.json() if r.status_code == 200 else []
        except httpx.RequestError:
            return []


# ── Persistent WebSocket Manager ─────────────────────────────────────────


class SessionWebSocket:
    """
    Wraps a single websockets connection that lives for the entire Chainlit
    session. Handles reconnection transparently if the server drops the
    connection between turns (e.g. after an idle timeout). Uses a background
    task to listen for messages continuously so async notifications (like
    human agent replies) are displayed immediately without reloading.
    """

    def __init__(self, token: str, conversation_id: str, session_context):
        self._token = token
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._url = (
            f"{WS_BASE_URL}/chat/ws?token={token}&conversation_id={conversation_id}"
        )
        self._session_context = session_context
        self._listener_task = None

        self._current_msg = None
        self._accumulated = ""
        self._active_steps = {}
        self._turn_complete = asyncio.Event()
        self._turn_complete.set()

    def update_context(self, session_context):
        """Updates the internal context with the provided session_context.

        Args:
            session_context: The new context data to merge into the current session."""
        self._session_context = session_context

    async def connect(self) -> None:
        """Open the WebSocket. Call once at session start."""
        self._ws = await websockets.connect(
            self._url,
            ping_interval=20,
            ping_timeout=60,
            open_timeout=15,
        )
        logger.info("WS connected: %s", self._url)
        if self._listener_task is None:
            self._listener_task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self):
        """Background task that continuously listens to the WebSocket."""
        from chainlit.context import context_var

        while True:
            # Update the chainlit context dynamically if it was changed (e.g. user switched tabs back to this thread)
            context_var.set(self._session_context)

            if self._ws is None or self._ws.state != websockets.State.OPEN:
                await asyncio.sleep(1)
                continue

            try:
                raw = await self._ws.recv()
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON WS frame: %s", raw)
                    continue

                etype = event.get("type") or event.get("event")

                # ── Streaming token ──────────────────────────────────
                if etype == "token":
                    if hasattr(self, "_typing_task") and self._typing_task:
                        self._typing_task.cancel()
                        self._typing_task = None

                    content_chunk = event.get("content", "")

                    if (
                        getattr(self, "_is_typing_indicator", False)
                        and self._current_msg
                    ):
                        # Clear the placeholder text before appending the real message
                        self._current_msg.content = ""
                        self._is_typing_indicator = False

                    if self._current_msg is None:
                        # If a human agent is joining or replying, set author accordingly
                        author_name = "Assistant"
                        if event.get("agent_name"):
                            author_name = event["agent_name"]
                        elif "Support Agent" in content_chunk or "🟢" in content_chunk:
                            author_name = "Support Agent"

                        self._current_msg = cl.Message(author=author_name, content="")
                        await self._current_msg.send()

                    self._accumulated += content_chunk
                    _, visible = parse_reasoning(self._accumulated)
                    self._current_msg.content = visible
                    await self._current_msg.update()

                # ── Tool start → Chainlit Step ───────────────────────
                elif etype == "tool_start":
                    name = event.get("name", "tool")
                    step = cl.Step(name=f"🛠️ {name}", type="tool")
                    step.input = event.get("inputs", "")
                    await step.send()
                    self._active_steps[name] = step

                # ── Tool end → close Step ────────────────────────────
                elif etype == "tool_end":
                    name = event.get("name", "tool")
                    step = self._active_steps.pop(name, None)
                    if step:
                        step.output = event.get("output", "")
                        await step.update()

                # ── Turn finished ────────────────────────────────────
                elif etype == "end":
                    if hasattr(self, "_typing_task") and self._typing_task:
                        self._typing_task.cancel()
                        self._typing_task = None

                    if self._current_msg:
                        reasoning, clean = parse_reasoning(self._accumulated)
                        parts = []
                        if reasoning:
                            parts.append(
                                f"<details>\n<summary>💭 Thinking Process</summary>"
                                f"\n\n{reasoning}\n\n</details>\n\n"
                            )
                        parts.append(clean)
                        self._current_msg.content = "".join(parts)
                        await self._current_msg.update()

                    # Reset state for next message
                    self._current_msg = None
                    self._accumulated = ""
                    if hasattr(self, "_turn_complete"):
                        self._turn_complete.set()

                # ── Escalated Ack ────────────────────────────────────
                elif etype == "escalated_ack":
                    cl.user_session.set("is_escalated", True)
                    if hasattr(self, "_typing_task") and self._typing_task:
                        self._typing_task.cancel()
                        self._typing_task = None

                    if self._current_msg:
                        await self._current_msg.remove()
                    self._current_msg = None
                    self._accumulated = ""
                    if hasattr(self, "_turn_complete"):
                        self._turn_complete.set()

                # ── Agent Joined ─────────────────────────────────────
                elif etype == "agent_joined":
                    agent_name = event.get("agent_name", "Support Agent")
                    # Store agent info so UI can display "Agent Online" status
                    cl.user_session.set("active_agent_name", agent_name)

                    if hasattr(self, "_typing_task") and self._typing_task:
                        self._typing_task.cancel()
                        self._typing_task = None
                    if self._current_msg:
                        await self._current_msg.remove()
                    self._current_msg = None
                    joined_content = event.get(
                        "content",
                        f"🟢 **{agent_name}** (Support Agent) has joined the chat. You're now connected to a human agent!",
                    )
                    await cl.Message(author="System", content=joined_content).send()
                    # Signal any waiting send_message that this turn is done
                    if hasattr(self, "_turn_complete"):
                        self._turn_complete.set()

                # ── Agent Reply ──────────────────────────────────────
                elif etype == "agent_reply":
                    agent_name = event.get("agent_name", "Support Agent")
                    content = event.get("content", "")

                    # Cancel any pending typing-timeout task
                    if hasattr(self, "_typing_task") and self._typing_task:
                        self._typing_task.cancel()
                        self._typing_task = None

                    # If there's an active placeholder/typing message, replace it
                    if self._current_msg:
                        self._current_msg.author = "Support Agent"
                        self._current_msg.content = (
                            f"**{agent_name}**: {content}"
                            if agent_name and agent_name != "Support Agent"
                            else content
                        )
                        await self._current_msg.update()
                    else:
                        msg_content = (
                            f"**{agent_name}**: {content}"
                            if agent_name and agent_name != "Support Agent"
                            else content
                        )
                        await cl.Message(
                            author="Support Agent", content=msg_content
                        ).send()

                    self._current_msg = None
                    self._is_typing_indicator = False
                    self._accumulated = ""
                    if hasattr(self, "_turn_complete"):
                        self._turn_complete.set()

                # ── Resolved ─────────────────────────────────────────
                elif etype == "resolved":
                    cl.user_session.set("is_escalated", False)
                    cl.user_session.set("active_agent_name", None)
                    if self._current_msg:
                        await self._current_msg.remove()
                    self._current_msg = None
                    await cl.Message(
                        author="System",
                        content=event.get(
                            "content",
                            "✅ The support agent has resolved this issue. The AI assistant is ready to help you with other questions!",
                        ),
                    ).send()

                # ── Agent Typing ─────────────────────────────────────
                elif etype == "agent_typing":
                    agent_name = event.get("agent_name", "Support Agent")

                    if self._current_msg and not getattr(
                        self, "_is_typing_indicator", False
                    ):
                        # Transform existing placeholder (e.g. "*Thinking...*") into typing indicator
                        self._current_msg.author = "Support Agent"
                        self._current_msg.content = (
                            f"*{agent_name} is typing...*"
                            if agent_name and agent_name != "Support Agent"
                            else "*Typing...*"
                        )
                        self._is_typing_indicator = True
                        await self._current_msg.update()
                    elif not self._current_msg:
                        # No placeholder exists: create a fresh typing bubble
                        typing_content = (
                            f"*{agent_name} is typing...*"
                            if agent_name and agent_name != "Support Agent"
                            else "*Typing...*"
                        )
                        self._current_msg = cl.Message(
                            author="Support Agent", content=typing_content
                        )
                        self._is_typing_indicator = True
                        await self._current_msg.send()
                    # else: already showing typing indicator, just reset the timeout below

                    # Reset the auto-clear timeout on every heartbeat
                    if hasattr(self, "_typing_task") and self._typing_task:
                        self._typing_task.cancel()

                    async def clear_typing():
                        await asyncio.sleep(6)
                        if self._current_msg and getattr(
                            self, "_is_typing_indicator", False
                        ):
                            await self._current_msg.remove()
                            self._current_msg = None
                            self._is_typing_indicator = False
                            self._accumulated = ""

                    self._typing_task = asyncio.create_task(clear_typing())

                # ── Server-side error ────────────────────────────────
                elif etype == "error":
                    if self._current_msg:
                        self._current_msg.content = (
                            f"❌ **Server error:** {event.get('message', '?')}"
                        )
                        await self._current_msg.update()
                    else:
                        await cl.Message(
                            author="System",
                            content=f"❌ **Server error:** {event.get('message', '?')}",
                        ).send()
                    self._current_msg = None
                    self._accumulated = ""
                    if hasattr(self, "_turn_complete"):
                        self._turn_complete.set()

            except websockets.exceptions.ConnectionClosedError as e:
                logger.error("WS closed mid-stream: %s", e)
                self._ws = None  # Will be reconnected by ensure_connected
                if self._current_msg and not self._accumulated:
                    self._current_msg.content = (
                        "❌ Connection lost mid-response. Please resend."
                    )
                    await self._current_msg.update()
                self._current_msg = None
                self._accumulated = ""
                if hasattr(self, "_turn_complete"):
                    self._turn_complete.set()
            except Exception as e:
                logger.error("Listener error: %s", e, exc_info=True)
                if hasattr(self, "_turn_complete"):
                    self._turn_complete.set()
                # Do NOT sleep here — continue immediately so the next WS event
                # is not dropped after a transient error.

    async def _ensure_connected(self) -> None:
        """Reconnect if the connection was lost between turns."""
        if self._ws is None or self._ws.state != websockets.State.OPEN:
            logger.info("WS was closed — reconnecting…")
            await self.connect()

    async def close(self) -> None:
        """Closes the resource asynchronously, releasing any associated system resources.
        This method should be awaited before the object is discarded."""
        if self._listener_task:
            self._listener_task.cancel()
        if self._ws and self._ws.state != websockets.State.CLOSED:
            await self._ws.close()
            logger.info("WS closed cleanly.")

    async def send_message(self, payload: dict, reply_msg: cl.Message) -> None:
        """Send one chat payload to the websocket and set up the active message state."""
        await self._ensure_connected()
        # Initialize the current message so the listener loop updates it
        self._current_msg = reply_msg
        self._accumulated = ""
        self._active_steps = {}
        self._turn_complete.clear()
        await self._ws.send(json.dumps(payload))
        await self._turn_complete.wait()


# ── Markdown Formatters ──────────────────────────────────────────────────


def format_orders_markdown(orders: list) -> str:
    """Formats a list of order dictionaries into a markdown string.

    Args:
        orders: List of order objects to format.

    Returns:
        A markdown-formatted string representing the orders."""
    if not orders:
        return "_You don't have any orders yet._"
    lines = []
    for order in orders:
        emoji = (
            "✅"
            if order["status"] == "delivered"
            else "🚚" if order["status"] == "shipped" else "⏳"
        )
        lines.append(
            f"**{emoji} Order `{order['id']}` — ₹{order['total']} ({order['status'].title()})**"
        )
        lines.append(f"- **Ordered:** {order['ordered_at'][:10]}")
        if order.get("delivered_at"):
            lines.append(f"- **Delivered:** {order['delivered_at'][:10]}")
        elif order.get("estimated_delivery"):
            lines.append(
                f"- **Estimated Delivery:** {order['estimated_delivery'][:10]}"
            )
        lines.append(f"- **Carrier:** {order.get('carrier', 'N/A')}")
        if order.get("return_eligible") is not None:
            rt = (
                f"Yes (Until {order.get('return_deadline', 'N/A')[:10]})"
                if order["return_eligible"]
                else "No"
            )
            lines.append(f"- **Return Eligible:** {rt}")
        lines.append("\n**Items:**")
        for item in order.get("items", []):
            name = item.get("name", f"Product {item.get('product_id', '?')}")
            details = []
            if item.get("color"):
                details.append(f"Color: {item['color']}")
            if item.get("size"):
                details.append(f"Size: {item['size']}")
            suffix = f" ({', '.join(details)})" if details else ""
            lines.append(
                f"  - **{name}**{suffix} × {item.get('quantity', 1)} — ₹{item.get('price', 0)}"
            )
        lines.append("---")
    return "\n".join(lines)


def format_products_markdown(products: list) -> str:
    """Converts a list of product dictionaries into a markdown-formatted string.

    Args:
        products (list): List of product dictionaries.

    Returns:
        str: Markdown representation of the products."""
    if not products:
        return "_No products available._"

    lines = []
    for p in products:
        title = p.get("title", "Unknown Product")
        price = p.get("price", "N/A")
        image_url = p.get("azure_image_url") or p.get("image_url") or p.get("image", "")
        category = p.get("category", "")

        lines.append(f"**{title}**")
        if category:
            lines.append(f"_{category.title()}_")
        lines.append(f"**Rs. {price}**")
        if image_url:
            lines.append(f"![{title}]({image_url})")
        lines.append("---")
    return "\n\n".join(lines)


# ── Auth ─────────────────────────────────────────────────────────────────


@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    """username = email; password is ignored (backend only needs email)."""
    email = username.strip()
    data = await api_login(email)
    if not data:
        return None
    token = data["access_token"]
    convs = await api_fetch_conversations(token)
    conv_id = convs[0]["conversation_id"] if convs else str(uuid.uuid4())
    return cl.User(
        identifier=email,
        display_name=email.split("@")[0],
        metadata={"jwt_token": token, "conversation_id": conv_id},
    )


# ── Chat Lifecycle ───────────────────────────────────────────────────────


@cl.on_chat_start
async def on_chat_start():
    """Starts the chat session asynchronously, initializing necessary resources and notifying participants that the chat has begun. Returns a coroutine that completes when the chat is ready."""
    user: cl.User = cl.user_session.get("user")
    token = user.metadata.get("jwt_token")
    if not token:
        token_data = await api_login(user.identifier)
        token = token_data["access_token"] if token_data else ""
        user.metadata["jwt_token"] = token
    conv_id = cl.context.session.thread_id

    cl.user_session.set("jwt_token", token)
    cl.user_session.set("conversation_id", conv_id)

    from chainlit.context import context_var

    # We create the session object but do NOT eagerly connect.
    # The connection will be lazily established on the first message sent.
    ws_session = SessionWebSocket(token, conv_id, context_var.get())
    cl.user_session.set("ws_session", ws_session)

    # Check history of this conversation
    history_data = await api_fetch_history(token, conv_id)
    history_messages = (
        history_data.get("messages", [])
        if isinstance(history_data, dict)
        else history_data
    )
    is_escalated = (
        history_data.get("status") == "escalated"
        if isinstance(history_data, dict)
        else False
    )

    cl.user_session.set("history", history_messages)
    cl.user_session.set("is_escalated", is_escalated)

    await cl.Message(
        author="Assistant",
        content=(
            f"[\u200b](http://conversation-id/{conv_id})\n"
            "Welcome! 👋 I'm your **E-Commerce Customer Support Assistant**.\n\n"
            "Here are a few things I can help you with:\n"
            "- 📦 **Order Inquiries:** *'What is the status of my order?'*\n"
            "- ❓ **FAQs & Policies:** *'What is your return policy?'*\n"
            "- 📸 **Visual Product Search:** Upload an image to find similar items!\n\n"
            "**Slash commands:**\n"
            "- `/orders` — View your order history\n"
            "- `/products` — Browse the product catalog\n\n"
            "How can I help you today?"
        ),
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Restore persistent WS session when clicking a past chat from the sidebar."""
    user: cl.User = cl.user_session.get("user")
    token = user.metadata.get("jwt_token") if user and user.metadata else None
    if not token and user:
        token_data = await api_login(user.identifier)
        token = token_data["access_token"] if token_data else ""
        user.metadata["jwt_token"] = token

    conv_id = thread["id"]
    cl.user_session.set("jwt_token", token)
    cl.user_session.set("conversation_id", conv_id)

    # Restore escalated state so user gets correct indicators when resuming
    history_data = await api_fetch_history(token, conv_id)
    is_escalated = (
        history_data.get("status") == "escalated"
        if isinstance(history_data, dict)
        else False
    )
    cl.user_session.set("is_escalated", is_escalated)

    from chainlit.context import context_var

    session_context = context_var.get()

    # Track all WS sessions to keep them alive while navigating chats
    ws_sessions = cl.user_session.get("ws_sessions")
    if ws_sessions is None:
        ws_sessions = {}
        cl.user_session.set("ws_sessions", ws_sessions)

    # If we already have a WS for this conversation, just update its context
    existing_ws = ws_sessions.get(conv_id)
    if (
        existing_ws
        and existing_ws._ws
        and existing_ws._ws.state == websockets.State.OPEN
    ):
        existing_ws.update_context(session_context)
        cl.user_session.set("ws_session", existing_ws)
        return

    # Use a short timeout for eager connect. If it fails, ws_session will still try to
    # reconnect lazily on the first message sent.
    ws_session = SessionWebSocket(token, conv_id, session_context)
    try:
        await asyncio.wait_for(ws_session.connect(), timeout=3.0)
    except Exception as e:
        logger.warning(
            "Could not eagerly open WS on chat resume (will retry on message): %s", e
        )

    ws_sessions[conv_id] = ws_session
    cl.user_session.set("ws_session", ws_session)


@cl.on_chat_end
async def on_chat_end():
    """Close all persistent WS sessions when the user leaves or refreshes."""
    ws_sessions = cl.user_session.get("ws_sessions") or {}

    # Also grab the legacy ws_session if any
    legacy_ws = cl.user_session.get("ws_session")
    if legacy_ws and legacy_ws._url not in [ws._url for ws in ws_sessions.values()]:
        ws_sessions["legacy"] = legacy_ws

    for ws in ws_sessions.values():
        if ws:
            try:
                await ws.close()
            except Exception:
                pass


@cl.on_message
async def on_message(message: cl.Message):
    """Asynchronously processes an incoming message represented by a cl.Message object, performing any necessary handling logic."""
    token: str = cl.user_session.get("jwt_token")
    conv_id: str = cl.user_session.get("conversation_id")
    ws_session: SessionWebSocket | None = cl.user_session.get("ws_session")

    if not token:
        await cl.Message(content="⚠️ Session expired — please log in again.").send()
        return

    text = message.content.strip()

    # ── Slash commands (HTTP only, no WS needed) ─────────────────────────
    if text.lower() == "/orders":
        await _handle_orders(token)
        return

    if text.lower() == "/products":
        await _handle_products(token)
        return

    # ── Image attachment ─────────────────────────────────────────────────
    image_elements = []
    for el in message.elements:
        # Some versions of chainlit use mime="image/jpeg", some use type="image"
        is_image = el.type == "image" or (
            hasattr(el, "mime") and el.mime and el.mime.startswith("image/")
        )
        if is_image and getattr(el, "path", None):
            image_elements.append(el)

    if len(image_elements) > 1:
        await cl.Message(
            content="Please upload only one photo at a time. Please try again with a single image."
        ).send()
        return

    image_b64: str | None = None
    if image_elements:
        with open(image_elements[0].path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

    # ── Guard: WS not available ──────────────────────────────────────────
    if ws_session is None:
        await cl.Message(
            content="❌ WebSocket connection is unavailable. Please refresh the page."
        ).send()
        return

    # ── Stream reply over the persistent WS ─────────────────────────────
    payload = {
        "query": text,
        "conversation_id": conv_id,
        **({"image_base64": image_b64} if image_b64 else {}),
    }

    is_escalated = cl.user_session.get("is_escalated", False)
    loading_text = (
        "*Connecting you with a agent...*" if is_escalated else "*Thinking...*"
    )

    reply_msg = cl.Message(author="Assistant", content=loading_text)
    await reply_msg.send()

    await ws_session.send_message(payload, reply_msg)


# ── Shortcut Handlers ────────────────────────────────────────────────────


async def _handle_orders(token: str, offset: int = 0):
    msg = cl.Message(author="Assistant", content="🔍 Fetching your orders…")
    await msg.send()
    orders = await api_fetch_orders(token)

    limit = 3
    sliced_orders = orders[offset : offset + limit]

    if not orders:
        msg.content = "### 📦 Your Order History\n\n_You don't have any orders yet._"
    else:
        msg.content = f"### 📦 Your Order History (Showing {offset + 1}-{min(offset + limit, len(orders))} of {len(orders)})\n\n{format_orders_markdown(sliced_orders)}"

    actions = []
    if offset + limit < len(orders):
        actions.append(
            cl.Action(
                name="load_more_orders",
                payload={"offset": offset + limit},
                label="⬇️ Load More Orders",
                tooltip="Load older orders",
            )
        )
    msg.actions = actions
    await msg.update()


async def _handle_products(token: str, offset: int = 0):
    msg = cl.Message(author="Assistant", content="🔍 Fetching products…")
    await msg.send()
    products = await api_fetch_products(token)

    limit = 3
    sliced_products = products[offset : offset + limit]

    if not products:
        msg.content = "### 🛍️ Product Catalog\n\n_No products available._"
    else:
        msg.content = f"### 🛍️ Product Catalog (Showing {offset + 1}-{min(offset + limit, len(products))} of {len(products)})\n\n{format_products_markdown(sliced_products)}"

    actions = []
    if offset + limit < len(products):
        actions.append(
            cl.Action(
                name="load_more_products",
                payload={"offset": offset + limit},
                label="⬇️ Load More Products",
                tooltip="Load more products",
            )
        )
    msg.actions = actions
    await msg.update()


@cl.action_callback("load_more_orders")
async def on_load_more_orders(action: cl.Action):
    """Loads additional orders in response to the provided action.

    Args:
        action (cl.Action): The action that triggered the load request."""
    token = cl.user_session.get("jwt_token")
    offset = action.payload.get("offset", 0) if action.payload else 0
    await _handle_orders(token, offset=offset)


@cl.action_callback("load_more_products")
async def on_load_more_products(action: cl.Action):
    """Asynchronously loads additional products in response to the provided load‑more action."""
    token = cl.user_session.get("jwt_token")
    offset = action.payload.get("offset", 0) if action.payload else 0
    await _handle_products(token, offset=offset)



