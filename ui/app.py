"""
Streamlit Chat UI for E-Commerce Customer Assist.

Connects to the FastAPI backend for authentication and chat.
"""

import os
import uuid
import streamlit as st
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Support Chat",
    page_icon="🛒",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Chat container styling */
    .stChatMessage {
        border-radius: 12px;
    }

    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .login-success {
        padding: 0.75rem;
        border-radius: 8px;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session State Initialization ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None


# ── Helper Functions ─────────────────────────────────────────────────────


def login(email: str) -> bool:
    """Authenticate against the backend and store the JWT token."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/auth/login",
            json={"email": email},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.jwt_token = data["access_token"]
            st.session_state.user_email = email
            
            # Load conversations, if any exist pick the most recent, else make new
            convs = fetch_conversations()
            if convs:
                st.session_state.conversation_id = convs[0]["conversation_id"]
                st.session_state.messages = fetch_history(st.session_state.conversation_id)
            else:
                st.session_state.conversation_id = str(uuid.uuid4())
            return True
        else:
            return False
    except requests.RequestException:
        return False


def send_message(query: str) -> str | None:
    """Send a chat message to the backend and return the AI response text."""
    if not st.session_state.jwt_token:
        return None

    headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
    payload = {"query": query}

    if st.session_state.conversation_id:
        payload["conversation_id"] = st.session_state.conversation_id

    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            headers=headers,
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            return extract_response_text(data)
        else:
            return f"Error: Server returned status {resp.status_code}."
    except requests.RequestException as e:
        return f"Error: Could not connect to the server. ({e})"


def extract_response_text(data: dict) -> str:
    """Extract the human-readable response text from the graph state."""

    # Try agent_response first
    agent_resp = data.get("agent_response")
    if agent_resp and isinstance(agent_resp, dict):
        text = agent_resp.get("resolution_text")
        if text:
            return text

    # Fall back to messages — find the last AI message
    messages = data.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, dict):
            msg_type = msg.get("type", "")
            if msg_type == "ai" and msg.get("content"):
                return msg["content"]
        elif isinstance(msg, list):
            # Handle serialized message format
            for item in reversed(msg):
                if isinstance(item, dict) and item.get("type") == "ai":
                    return item.get("content", "")

    return "I received your message but couldn't generate a response. Please try again."


def fetch_conversations():
    if not st.session_state.jwt_token: return []
    try:
        resp = requests.get(
            f"{API_BASE_URL}/chat/conversations",
            headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []

def fetch_history(conversation_id):
    if not st.session_state.jwt_token: return []
    try:
        resp = requests.get(
            f"{API_BASE_URL}/chat/history/{conversation_id}",
            headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json().get("messages", [])
    except Exception:
        pass
    return []

def logout():
    """Clear session state and log the user out."""
    st.session_state.jwt_token = None
    st.session_state.user_email = None
    st.session_state.messages = []
    st.session_state.conversation_id = None


# ── Sidebar: Authentication ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔐 Authentication")
    st.divider()

    if st.session_state.jwt_token:
        st.success(f"Logged in as **{st.session_state.user_email}**")

        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()

        st.divider()
        st.markdown("### 💬 Recent Chats")

        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

        conversations = fetch_conversations()
        for conv in conversations:
            title = conv.get("title", "Chat")
            conv_id = conv.get("conversation_id")
            
            btn_type = "primary" if conv_id == st.session_state.conversation_id else "secondary"
            if st.button(f"📄 {title}", key=f"btn_{conv_id}", type=btn_type, use_container_width=True):
                st.session_state.conversation_id = conv_id
                st.session_state.messages = fetch_history(conv_id)
                st.rerun()

        st.divider()
        if st.button("🗑️ Clear Local Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    else:
        st.info("Please log in to start chatting.")
        email = st.text_input(
            "Email Address",
            placeholder="e.g. customer@example.com",
        )

        if st.button("🔑 Login", use_container_width=True):
            if email:
                with st.spinner("Authenticating..."):
                    if login(email):
                        st.rerun()
                    else:
                        st.error("Login failed. Please check your email and try again.")
            else:
                st.warning("Please enter your email address.")


# ── Main Chat Area ───────────────────────────────────────────────────────
st.title("🛒 Customer Support Chat")

if not st.session_state.jwt_token:
    st.markdown(
        """
        Welcome to the **E-Commerce Customer Support Assistant**!

        I can help you with:
        - 📦 **Order inquiries** — track orders, check details, find past purchases
        - ❓ **FAQs** — return policies, shipping info, company details

        👈 **Please log in from the sidebar to get started.**
        """
    )
else:
    # Display existing chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about your orders or our policies..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = send_message(prompt)

            if response_text:
                st.markdown(response_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )
            else:
                error_msg = "Something went wrong. Please try again."
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
