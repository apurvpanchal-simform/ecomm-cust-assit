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

    /* Add bottom padding to ensure the last messages scroll above the fixed footer */
    .stMain .block-container {
        padding-bottom: 220px !important;
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
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False


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


def send_message(query: str, image_b64: str = None) -> str | None:
    """Send a chat message to the backend and return the AI response text."""
    if not st.session_state.jwt_token:
        return None

    headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
    payload = {"query": query}
    
    if image_b64:
        payload["image_base64"] = image_b64

    if st.session_state.conversation_id:
        payload["conversation_id"] = st.session_state.conversation_id

    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            headers=headers,
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            return extract_response_text(data)
        else:
            return f"Error: Server returned status {resp.status_code}."
    except requests.RequestException as e:
        return f"Error: Could not connect to the server. ({e})"


def extract_response_text(data: dict) -> str:
    """Extract the human-readable response text from the graph state.

    We prioritize the last AI message from the messages list because
    `agent_response` is a plain dict in the state that persists across
    checkpointed turns and can return stale answers from a previous run.
    """

    # Primary: find the last AI message (always the freshest response)
    messages = data.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, dict):
            msg_type = msg.get("type", "")
            if msg_type == "ai" and msg.get("content"):
                return msg["content"]
        elif isinstance(msg, list):
            for item in reversed(msg):
                if isinstance(item, dict) and item.get("type") == "ai":
                    return item.get("content", "")

    # Fallback: try agent_response
    agent_resp = data.get("agent_response")
    if agent_resp and isinstance(agent_resp, dict):
        text = agent_resp.get("resolution_text")
        if text:
            return text

    return "I received your message but couldn't generate a response. Please try again."


def parse_reasoning(content: str) -> tuple[str | None, str]:
    """Extracts reasoning (inside <think> tags) and the clean response content."""
    import re
    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return reasoning, clean_content
    return None, content


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

def delete_conversation(conversation_id):
    if not st.session_state.jwt_token: return False
    try:
        resp = requests.delete(
            f"{API_BASE_URL}/chat/conversations/{conversation_id}",
            headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
            timeout=10
        )
        return resp.status_code == 200
    except Exception:
        return False


@st.dialog("📦 Your Order History", width="large")
def show_orders_dialog():
    if not st.session_state.jwt_token: return
    try:
        with st.spinner("Fetching your orders..."):
            resp = requests.get(
                f"{API_BASE_URL}/orders",
                headers={"Authorization": f"Bearer {st.session_state.jwt_token}"},
                timeout=10
            )
        if resp.status_code == 200:
            orders = resp.json()
            if not orders:
                st.info("You don't have any orders yet.")
                return
                
            for order in orders:
                status_emoji = "✅" if order["status"] == "delivered" else "🚚" if order["status"] == "shipped" else "⏳"
                with st.expander(f"{status_emoji} Order **{order['id']}** - ${order['total']} ({order['status'].title()})"):
                    st.write(f"**Ordered:** {order['ordered_at'][:10]}")
                    if order.get("delivered_at"):
                        st.write(f"**Delivered:** {order['delivered_at'][:10]}")
                    elif order.get("estimated_delivery"):
                        st.write(f"**Estimated Delivery:** {order['estimated_delivery'][:10]}")
                    
                    st.write(f"**Carrier:** {order.get('carrier', 'N/A')}")
                    
                    if order.get('return_eligible') is not None:
                        return_text = f"Yes (Until {order.get('return_deadline', 'N/A')[:10]})" if order.get('return_eligible') else "No"
                        st.write(f"**Return Eligible:** {return_text}")
                    
                    st.markdown("#### Items")
                    for item in order.get("items", []):
                        item_name = item.get('name', f"Product {item.get('product_id', 'Unknown')}")
                        details = []
                        if item.get("color"):
                            details.append(f"Color: {item['color']}")
                        if item.get("size"):
                            details.append(f"Size: {item['size']}")
                        
                        detail_str = f" ({', '.join(details)})" if details else ""
                        st.markdown(f"- **{item_name}**{detail_str} (x{item.get('quantity', 1)}) - ${item.get('price', 0)}")
        else:
            st.error("Failed to load orders. Please try again.")
    except Exception as e:
        st.error(f"Error connecting to server: {e}")

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
            st.session_state.uploader_key += 1
            st.rerun()

        conversations = fetch_conversations()
        for conv in conversations:
            title = conv.get("title", "Chat")
            conv_id = conv.get("conversation_id")
            
            btn_type = "primary" if conv_id == st.session_state.conversation_id else "secondary"
            
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(f"📄 {title}", key=f"btn_{conv_id}", type=btn_type, use_container_width=True):
                    st.session_state.conversation_id = conv_id
                    st.session_state.messages = fetch_history(conv_id)
                    st.session_state.uploader_key += 1
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{conv_id}", help="Delete chat", use_container_width=True):
                    if delete_conversation(conv_id):
                        if st.session_state.conversation_id == conv_id:
                            st.session_state.conversation_id = str(uuid.uuid4())
                            st.session_state.messages = []
                        st.rerun()

        st.divider()
        st.markdown("### 🚀 Shortcuts")
        
        if st.button("📦 Show My Orders", use_container_width=True):
            show_orders_dialog()

        st.divider()
        if st.button("🗑️ Clear Local Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploader_key += 1
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
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Welcome! 👋 I'm your **E-Commerce Customer Support Assistant**.\n\n"
                "Here are a few things I can help you with:\n"
                "- 📦 **Order Inquiries:** *'What is the status of my order?'* or *'Show me my recent purchases.'*\n"
                "- ❓ **FAQs & Policies:** *'What is your return policy?'* or *'How long does shipping take?'*\n"
                "- 📸 **Visual Product Search:** *Upload an image of a product to find similar items in our catalog!*\n\n"
                "How can I help you today?"
            )
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    reasoning, clean_text = parse_reasoning(message["content"])
                    if reasoning:
                        with st.expander("💭 Thinking Process"):
                            st.markdown(reasoning)
                    st.markdown(clean_text, unsafe_allow_html=True)
                else:
                    st.markdown(message["content"], unsafe_allow_html=True)
                    if message.get("image"):
                        import base64
                        st.image(base64.b64decode(message["image"]), width=120)

        # 1. Anchor and floating scroll-to-bottom button
        st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="position: fixed; bottom: 85px; right: 50px; z-index: 99999;">
                <a href="#chat-bottom" target="_self">
                    <button style="
                        border-radius: 50%;
                        width: 40px;
                        height: 40px;
                        background-color: #f0f2f6;
                        color: #31333f;
                        border: 1px solid #ddd;
                        font-size: 18px;
                        cursor: pointer;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.15);
                    ">↓</button>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 1. Render file uploader above if toggled
    uploaded_file = None
    if st.session_state.show_uploader:
        uploaded_file = st.file_uploader(
            "Upload an image:",
            type=["png", "jpg", "jpeg"],
            key=f"uploader_{st.session_state.uploader_key}"
        )
        if uploaded_file:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("📤 Send Image without Text", use_container_width=True):
                    st.session_state.shortcut = "Can you find products similar to this image?"
                    st.rerun()

    # 2. Chat input and toggle button side-by-side (button on right, vertically centered)
    if st.button("➕", key="toggle_uploader"):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()

    prompt = st.chat_input("Ask me about your orders or our policies...")
    
    # Check for shortcut trigger
    if getattr(st.session_state, 'shortcut', None):
        prompt = st.session_state.shortcut
        st.session_state.shortcut = None

    # ── Dynamic Layout Adjustment via Javascript ──────────────────────────────
    st.components.v1.html(
        """
        <script>
        const doc = window.parent.document;
        const adjustLayout = () => {
            const chatInput = doc.querySelector('div[data-testid="stChatInput"]');
            const buttons = doc.querySelectorAll('.stMain div[data-testid="stButton"]');
            
            let buttonContainer = null;
            for (const container of buttons) {
                const btn = container.querySelector('button');
                if (btn && btn.textContent.trim() === '➕') {
                    buttonContainer = container;
                    break;
                }
            }
            
            if (chatInput && buttonContainer) {
                // Adjust chat input to make space on the right for the button
                chatInput.style.marginRight = '52px';
                chatInput.style.marginLeft = '0px';
                chatInput.style.width = 'calc(100% - 52px)';
                
                const rect = chatInput.getBoundingClientRect();
                
                // Position and style the '+' button container on the right
                buttonContainer.style.position = 'fixed';
                buttonContainer.style.zIndex = '99999';
                buttonContainer.style.left = `${rect.right + 8}px`;
                buttonContainer.style.top = `${rect.top + (rect.height - 40) / 2}px`;
                buttonContainer.style.margin = '0';
                buttonContainer.style.padding = '0';
                buttonContainer.style.display = 'block';
                
                // Style the button element itself
                const btn = buttonContainer.querySelector('button');
                if (btn) {
                    btn.style.height = '40px';
                    btn.style.width = '40px';
                    btn.style.borderRadius = '50%';
                    btn.style.padding = '0';
                    btn.style.margin = '0';
                    btn.style.display = 'flex';
                    btn.style.alignItems = 'center';
                    btn.style.justifyContent = 'center';
                    btn.style.fontSize = '1.2rem';
                    btn.style.lineHeight = '1';
                    
                    // Remove standard Streamlit button styling to make it an icon only
                    btn.style.background = 'transparent';
                    btn.style.border = 'none';
                    btn.style.boxShadow = 'none';
                    btn.style.color = 'inherit';
                }
            }
            
            // Align file uploader directly above chat input if visible
            const uploader = doc.querySelector('div[data-testid="element-container"]:has(div[data-testid="stFileUploader"])');
            if (uploader && chatInput) {
                const rect = chatInput.getBoundingClientRect();
                uploader.style.position = 'fixed';
                uploader.style.zIndex = '99998';
                uploader.style.left = `${rect.left}px`;
                uploader.style.width = `${rect.width}px`;
                uploader.style.bottom = `${window.parent.innerHeight - rect.top + 8}px`;
                uploader.style.transform = 'none';
                uploader.style.maxWidth = 'none';
                uploader.style.backgroundColor = 'transparent';
            }
        };
        
        // Run alignment on intervals to handle screen resizing, sidebar toggles, and state reruns
        setInterval(adjustLayout, 50);
        </script>
        """,
        height=0,
        width=0,
    )
    
    if prompt:
        actual_prompt = prompt
        
        image_b64 = None
        if uploaded_file:
            import base64
            image_b64 = base64.b64encode(uploaded_file.read()).decode("utf-8")
            
        # Add user message to history and display it
        user_msg = {"role": "user", "content": actual_prompt}
        if image_b64:
            user_msg["image"] = image_b64
        st.session_state.messages.append(user_msg)
        
        with st.chat_message("user"):
            st.markdown(actual_prompt)
            if uploaded_file:
                st.image(uploaded_file, width=120)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = send_message(actual_prompt, image_b64)

            if response_text:
                reasoning, clean_text = parse_reasoning(response_text)
                if reasoning:
                    with st.expander("💭 Thinking Process"):
                        st.markdown(reasoning)
                st.markdown(clean_text, unsafe_allow_html=True)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )
            else:
                error_msg = "Something went wrong. Please try again."
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
                
        if uploaded_file:
            st.session_state.uploader_key += 1
            st.session_state.show_uploader = False  # Hide it again after sending
            st.rerun()
