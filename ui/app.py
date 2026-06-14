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
from typing import Optional, Dict, List

import httpx
import websockets
import jwt
import psycopg
import chainlit as cl
import chainlit.data as cl_data
from chainlit.types import Pagination, ThreadFilter, PaginatedResponse, PageInfo, ThreadDict
from chainlit.user import User, PersistedUser
from chainlit.step import StepDict

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
WS_BASE_URL = API_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")


# ── Custom Data Layer Helpers ──────────────────────────────────────────────

async def get_customer_id_for_thread(thread_id: str) -> Optional[str]:
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        logger.error("SUPABASE_DB_URL is not set in environment.")
        return None
    try:
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT customer_id FROM customer_conversations WHERE conversation_id = %s",
                    (thread_id,)
                )
                row = await cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        logger.error(f"Failed to query customer_id for thread {thread_id}: {e}")
        return None

async def get_customer_email(customer_id: str) -> Optional[str]:
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url: return None
    try:
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT email FROM customers WHERE customer_id = %s", (customer_id,))
                row = await cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        logger.error(f"Failed to query email for customer {customer_id}: {e}")
        return None



def generate_system_jwt(customer_id: str) -> str:
    secret = os.getenv("JWT_SECRET_KEY")
    payload = {
        "sub": customer_id,
        "email": "system@ecomm-cust-assist.internal",
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


# ── Custom Data Layer for Persistence & History Sidebar ───────────────────

class CustomDataLayer(cl_data.BaseDataLayer):
    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        token_data = await api_login(identifier)
        token = token_data["access_token"] if token_data else ""
        return PersistedUser(
            id=identifier,
            createdAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            identifier=identifier,
            display_name=identifier.split("@")[0],
            metadata={"jwt_token": token}
        )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
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
            metadata=metadata
        )

    async def delete_feedback(self, feedback_id: str) -> bool:
        return True

    async def upsert_feedback(self, feedback) -> str:
        return "feedback_id"

    async def create_element(self, element):
        pass

    async def get_element(self, thread_id: str, element_id: str):
        return None

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        pass

    async def create_step(self, step_dict: StepDict):
        pass

    async def update_step(self, step_dict: StepDict):
        pass

    async def delete_step(self, step_id: str):
        pass

    async def get_thread_author(self, thread_id: str) -> str:
        customer_id = await get_customer_id_for_thread(thread_id)
        if not customer_id:
            return ""
        email = await get_customer_email(customer_id)
        return email or customer_id or ""

    async def delete_thread(self, thread_id: str):
        logger.info(f"[DataLayer] delete_thread called for thread_id={thread_id}")
        customer_id = await get_customer_id_for_thread(thread_id)
        if not customer_id:
            logger.warning(f"[DataLayer] Thread {thread_id} not found in DB, skipping deletion.")
            return

        token = generate_system_jwt(customer_id)
        await api_delete_conversation(token, thread_id)
        logger.info(f"[DataLayer] Thread {thread_id} deleted successfully.")

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        user_identifier = filters.userId
        if not user_identifier:
            return PaginatedResponse(pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None), data=[])

        token_data = await api_login(user_identifier)
        if not token_data:
            return PaginatedResponse(pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None), data=[])

        token = token_data["access_token"]
        convs = await api_fetch_conversations(token)

        threads = []
        for c in convs:
            threads.append({
                "id": c["conversation_id"],
                "createdAt": c["updated_at"] if isinstance(c.get("updated_at"), str) else datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "name": c["title"],
                "userId": user_identifier,
                "userIdentifier": user_identifier,
                "tags": [],
                "metadata": {},
                "steps": [],
                "elements": []
            })

        return PaginatedResponse(
            pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None),
            data=threads
        )

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        logger.info(f"[DataLayer] get_thread called for thread_id={thread_id}")
        user = None
        try:
            user = cl.user_session.get("user")
            logger.info(f"[DataLayer] user from cl.user_session: {user}")
        except Exception as e:
            logger.debug(f"[DataLayer] cl.user_session not available in get_thread: {e}")

        # Resolve customer_id from DB using thread_id
        customer_id = await get_customer_id_for_thread(thread_id)
        if not customer_id:
            logger.warning(f"[DataLayer] Thread {thread_id} not found in database customer_conversations")
            # If the thread is new and not in the DB yet, we can fallback to the current session user if available
            if user:
                token_data = await api_login(user.identifier)
                if token_data:
                    try:
                        payload = jwt.decode(token_data["access_token"], os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
                        customer_id = payload.get("sub")
                    except Exception as e:
                        logger.error(f"[DataLayer] Failed to decode fallback user token: {e}")

        if not customer_id:
            logger.warning(f"[DataLayer] No customer_id could be resolved for thread_id={thread_id}")
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
                "elements": []
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
            steps.append({
                "id": step_id,
                "threadId": thread_id,
                "parentId": None,
                "streaming": False,
                "metadata": {},
                "tags": [],
                "input": "",
                "isError": False,
                "output": content,
                "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "start": None,
                "end": None,
                "language": None,
                "name": "User" if role == "user" else "Assistant",
                "type": "user_message" if role == "user" else "assistant_message",
                "showInput": False,
            })

            for img_url in m.get("images", []):
                elements.append({
                    "id": str(uuid.uuid4()),
                    "threadId": thread_id,
                    "type": "image",
                    "url": img_url,
                    "name": "uploaded_image",
                    "display": "inline",
                    "forId": step_id
                })

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
            "elements": elements
        }

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        pass

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        pass

    async def get_favorite_steps(self, user_id: str) -> List[StepDict]:
        return []


@cl.data_layer
def get_data_layer():
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


# ── HTTP Helpers ─────────────────────────────────────────────────────────


async def api_login(email: str) -> dict | None:
    async with httpx.AsyncClient() as c:
        try:
            r = await c.post(f"{API_BASE_URL}/auth/login", json={"email": email}, timeout=10)
            return r.json() if r.status_code == 200 else None
        except httpx.RequestError:
            return None


async def api_fetch_conversations(token: str) -> list:
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
    connection between turns (e.g. after an idle timeout).
    """

    def __init__(self, token: str):
        self._token = token
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._lock = asyncio.Lock()          # one message in flight at a time
        self._url = f"{WS_BASE_URL}/chat/ws?token={token}"

    async def connect(self) -> None:
        """Open the WebSocket. Call once at session start."""
        self._ws = await websockets.connect(
            self._url,
            ping_interval=20,
            ping_timeout=60,
            open_timeout=15,
        )
        logger.info("WS connected: %s", self._url)

    async def _ensure_connected(self) -> None:
        """Reconnect if the connection was lost between turns."""
        if self._ws is None or self._ws.state != websockets.State.OPEN:
            logger.info("WS was closed — reconnecting…")
            await self.connect()

    async def close(self) -> None:
        if self._ws and self._ws.state != websockets.State.CLOSED:
            await self._ws.close()
            logger.info("WS closed cleanly.")

    async def stream_turn(
        self,
        payload: dict,
        reply_msg: cl.Message,
    ) -> None:
        """
        Send one chat payload and stream the response into reply_msg.
        Acquires the per-session lock so concurrent sends cannot interleave.
        """
        async with self._lock:
            await self._ensure_connected()
            await self._ws.send(json.dumps(payload))

            accumulated = ""
            active_steps: dict[str, cl.Step] = {}

            try:
                # ⚠️ Do NOT use `async for raw in self._ws:` here!
                # That iterator auto-closes the connection when the loop
                # exits (even via break), killing the persistent WS.
                # Use explicit recv() to keep the connection alive.
                while True:
                    raw = await self._ws.recv()
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON WS frame: %s", raw)
                        continue

                    etype = event.get("type")

                    # ── Streaming token ──────────────────────────────────
                    if etype == "token":
                        accumulated += event.get("content", "")
                        _, visible = parse_reasoning(accumulated)
                        reply_msg.content = visible
                        await reply_msg.update()

                    # ── Tool start → Chainlit Step ───────────────────────
                    elif etype == "tool_start":
                        name = event.get("name", "tool")
                        step = cl.Step(name=f"🛠️ {name}", type="tool")
                        step.input = event.get("inputs", "")
                        await step.send()
                        active_steps[name] = step

                    # ── Tool end → close Step ────────────────────────────
                    elif etype == "tool_end":
                        name = event.get("name", "tool")
                        step = active_steps.pop(name, None)
                        if step:
                            step.output = event.get("output", "")
                            await step.update()

                    # ── Turn finished ────────────────────────────────────
                    elif etype == "end":
                        reasoning, clean = parse_reasoning(accumulated)
                        parts = []
                        if reasoning:
                            parts.append(
                                f"<details>\n<summary>💭 Thinking Process</summary>"
                                f"\n\n{reasoning}\n\n</details>\n\n"
                            )
                        parts.append(clean)
                        reply_msg.content = "".join(parts)
                        await reply_msg.update()
                        # Break the recv() loop — WS stays open for next turn
                        break

                    # ── Server-side error ────────────────────────────────
                    elif etype == "error":
                        reply_msg.content = f"❌ **Server error:** {event.get('message', '?')}"
                        await reply_msg.update()
                        break

            except websockets.exceptions.ConnectionClosedError as e:
                logger.error("WS closed mid-stream: %s", e)
                self._ws = None          # force reconnect on next turn
                if not accumulated:
                    reply_msg.content = "❌ Connection lost mid-response. Please resend."
                    await reply_msg.update()


# ── Markdown Formatters ──────────────────────────────────────────────────


def format_orders_markdown(orders: list) -> str:
    if not orders:
        return "_You don't have any orders yet._"
    lines = []
    for order in orders:
        emoji = "✅" if order["status"] == "delivered" else "🚚" if order["status"] == "shipped" else "⏳"
        lines.append(f"**{emoji} Order `{order['id']}` — ₹{order['total']} ({order['status'].title()})**")
        lines.append(f"- **Ordered:** {order['ordered_at'][:10]}")
        if order.get("delivered_at"):
            lines.append(f"- **Delivered:** {order['delivered_at'][:10]}")
        elif order.get("estimated_delivery"):
            lines.append(f"- **Estimated Delivery:** {order['estimated_delivery'][:10]}")
        lines.append(f"- **Carrier:** {order.get('carrier', 'N/A')}")
        if order.get("return_eligible") is not None:
            rt = (
                f"Yes (Until {order.get('return_deadline', 'N/A')[:10]})"
                if order["return_eligible"] else "No"
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
            lines.append(f"  - **{name}**{suffix} × {item.get('quantity', 1)} — ₹{item.get('price', 0)}")
        lines.append("---")
    return "\n".join(lines)


def format_products_markdown(products: list) -> str:
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
    user: cl.User = cl.user_session.get("user")
    token = user.metadata.get("jwt_token")
    if not token:
        token_data = await api_login(user.identifier)
        token = token_data["access_token"] if token_data else ""
        user.metadata["jwt_token"] = token
    conv_id = cl.context.session.thread_id

    cl.user_session.set("jwt_token", token)
    cl.user_session.set("conversation_id", conv_id)

    # Open persistent WS connection for this session
    ws_session = SessionWebSocket(token)
    try:
        await ws_session.connect()
    except Exception as e:
        logger.error("Could not open WS on chat start: %s", e)
        # Store None; on_message will show an error instead of crashing
        ws_session = None
    cl.user_session.set("ws_session", ws_session)

    # Check history of this conversation
    history = await api_fetch_history(token, conv_id)
    cl.user_session.set("history", history)

    if not history:
        await cl.Message(
            author="Assistant",
            content=(
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

    ws_session = SessionWebSocket(token)
    try:
        await ws_session.connect()
    except Exception as e:
        logger.error("Could not open WS on chat resume: %s", e)
        ws_session = None
    cl.user_session.set("ws_session", ws_session)



@cl.on_chat_end
async def on_chat_end():
    """Close the persistent WS when the user leaves or refreshes."""
    ws_session: SessionWebSocket | None = cl.user_session.get("ws_session")
    if ws_session:
        await ws_session.close()


@cl.on_message
async def on_message(message: cl.Message):
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
    image_b64: str | None = None
    for el in message.elements:
        mime_type = getattr(el, "mime", "") or ""
        if "image" in mime_type and getattr(el, "path", None):
            with open(el.path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            break

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

    reply_msg = cl.Message(author="Assistant", content="*Thinking...*")
    await reply_msg.send()

    await ws_session.stream_turn(payload, reply_msg)


# ── Shortcut Handlers ────────────────────────────────────────────────────


async def _handle_orders(token: str, offset: int = 0):
    msg = cl.Message(author="Assistant", content="🔍 Fetching your orders…")
    await msg.send()
    orders = await api_fetch_orders(token)
    
    limit = 3
    sliced_orders = orders[offset:offset + limit]
    
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
                tooltip="Load older orders"
            )
        )
    msg.actions = actions
    await msg.update()


async def _handle_products(token: str, offset: int = 0):
    msg = cl.Message(author="Assistant", content="🔍 Fetching products…")
    await msg.send()
    products = await api_fetch_products(token)
    
    limit = 3
    sliced_products = products[offset:offset + limit]
    
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
                tooltip="Load more products"
            )
        )
    msg.actions = actions
    await msg.update()


@cl.action_callback("load_more_orders")
async def on_load_more_orders(action: cl.Action):
    token = cl.user_session.get("jwt_token")
    offset = action.payload.get("offset", 0) if action.payload else 0
    await _handle_orders(token, offset=offset)


@cl.action_callback("load_more_products")
async def on_load_more_products(action: cl.Action):
    token = cl.user_session.get("jwt_token")
    offset = action.payload.get("offset", 0) if action.payload else 0
    await _handle_products(token, offset=offset)

