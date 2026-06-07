<p align="center">
  <h1 align="center">🛒 E-Commerce Customer Assistant</h1>
  <p align="center">
    An AI-powered multi-agent customer support system with visual product search, built with LangGraph, FastAPI, and Streamlit.
  </p>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#license">License</a>
</p>

---

## Features

- **🤖 Multi-Agent Orchestration** — A supervisor agent intelligently routes user queries to specialized sub-agents (FAQ, Order, Visual Search) using LangGraph's state machine.
- **📦 Order Management** — Customers can check order status, view order details, track shipments, and look up past purchases using natural language.
- **❓ FAQ Knowledge Base** — RAG-powered FAQ agent retrieves answers from a curated knowledge base covering shipping, returns, payments, warranties, and account policies.
- **🖼️ Visual Product Search** — Upload a product image and the system uses OpenAI CLIP embeddings to find visually similar products from the catalog via Qdrant vector search.
- **💬 Persistent Chat Memory** — Dual-layer checkpointing system with Redis (hot cache) and Postgres (durable storage) ensures blazing-fast conversation recall with zero data loss.
- **📝 Automatic Summarization** — A dedicated summarizer agent condenses long conversation histories to keep context windows lean and inference costs low.
- **🔐 JWT Authentication** — Secure, stateless authentication via JSON Web Tokens tied to customer identities.
- **🔄 LLM Fallback Chain** — Primary model (Groq) with automatic fallback to Google Gemini, ensuring high availability.
- **📊 Token Usage Tracking** — Built-in callback handler that logs token consumption and estimated costs per request.
- **🚀 CI/CD Pipeline** — Fully automated GitHub Actions workflow that builds, containerizes, and deploys to Azure Container Apps on every push.

---

## Architecture

![alt text](ecomm_architecture.png)

### Agent Descriptions

| Agent | Purpose |
|---|---|
| **Supervisor** | Analyzes user intent and routes to the correct sub-agent(s). Handles greetings and chitchat directly. Supports multi-agent delegation for complex queries. |
| **FAQ** | Performs semantic search over the knowledge base (shipping, returns, payments, warranties, accounts) using Nomic/Gemini embeddings + Qdrant. |
| **Order** | Looks up customer-specific order data from Supabase using tool calls (`order_lookup`, `order_details`, `order_items`). |
| **CLIP Embedder** | Converts uploaded images or text queries into CLIP embedding vectors for visual similarity search. |
| **Visual Search** | Queries the `product_images` Qdrant collection to find the most visually similar products. |
| **Result Formatter** | Formats visual search results into a rich, human-readable response with product cards. |
| **Summarizer** | Compresses long conversation histories into concise summaries to stay within LLM context windows. |
| **Out of Domain** | Returns a friendly message when the user's query falls outside supported topics. |

---

## Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | Async REST API framework |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Multi-agent orchestration state machine |
| [LangChain](https://python.langchain.com/) | LLM abstractions, tools, and embeddings |
| [LangSmith](https://smith.langchain.com/) | Observability, tracing, and debugging |

### LLM Providers
| Provider | Model | Role |
|---|---|---|
| [Groq](https://groq.com/) | `qwen/qwen3-32b` | Primary inference (ultra-fast) |
| [Google Gemini](https://ai.google.dev/) | `gemini-2.5-flash-lite` | Fallback inference |

### Embeddings
| Provider | Model | Role |
|---|---|---|
| [Nomic](https://nomic.ai/) | `nomic-embed-text-v1.5` | Primary text embeddings (768d) |
| [Google Gemini](https://ai.google.dev/) | `gemini-embedding-2` | Fallback text embeddings (768d) |
| [OpenAI CLIP](https://github.com/mlfoundations/open_clip) | `ViT-B-32` | Image embeddings (512d) |

### Data Stores
| Service | Purpose |
|---|---|
| [Supabase](https://supabase.com/) (Postgres) | Customer data, orders, products, conversations, and durable LangGraph checkpoints |
| [Redis](https://redis.io/) | Hot-cache LangGraph checkpoints for fast conversation recall |
| [Qdrant Cloud](https://qdrant.tech/) | Vector database for FAQ semantic search and visual product search |
| [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs/) | Product image CDN |

### Frontend
| Technology | Purpose |
|---|---|
| [Streamlit](https://streamlit.io/) | Chat UI with image upload, conversation history, and rich product cards |

### DevOps
| Service | Purpose |
|---|---|
| [GitHub Actions](https://github.com/features/actions) | CI/CD pipeline |
| [Azure Container Registry](https://azure.microsoft.com/en-us/products/container-registry/) | Docker image storage |
| [Azure Container Apps](https://azure.microsoft.com/en-us/products/container-apps/) | Serverless container hosting |
| [Docker](https://www.docker.com/) | Multi-stage containerization |

---

## Project Structure

```
ecomm-cust-assit/
├── .github/
│   └── workflows/
│       └── azure-deploy.yml        # CI/CD pipeline
├── app/
│   ├── agents/                     # LangGraph agent nodes
│   │   ├── clip_embedding.py       # CLIP image/text embedding
│   │   ├── faq.py                  # FAQ retrieval agent
│   │   ├── order.py                # Order management agent
│   │   ├── result_formatter.py     # Visual search result formatter
│   │   ├── summarizer.py           # Conversation summarizer
│   │   ├── supervisor.py           # Intent router + orchestrator
│   │   └── visual_search.py        # Qdrant visual similarity search
│   ├── db/
│   │   ├── qdrant.py               # Qdrant client + collection setup
│   │   ├── schema.sql              # Supabase database schema
│   │   └── supabase.py             # Supabase client wrapper
│   ├── graph/
│   │   ├── builder.py              # LangGraph state machine definition
│   │   ├── checkpointer.py         # Dual Redis+Postgres checkpointer
│   │   └── state.py                # Shared AgentState TypedDict
│   ├── middleware/
│   │   └── auth.py                 # JWT authentication middleware
│   ├── schemas/
│   │   ├── agent.py                # Agent response schemas
│   │   └── api.py                  # API request/response models
│   ├── services/
│   │   ├── auth.py                 # JWT generation + validation
│   │   ├── azure_blob.py           # Azure Blob Storage upload
│   │   ├── clip_embedder.py        # CLIP model inference wrapper
│   │   ├── customers.py            # Customer lookup service
│   │   ├── llm.py                  # LLM + Embedding factory with fallbacks
│   │   ├── search.py               # Qdrant vector store wrapper
│   │   └── token_tracker.py        # Token usage + cost tracking
│   ├── tools/
│   │   ├── faq_search.py           # FAQ semantic search tool
│   │   ├── order_details.py        # Order details lookup tool
│   │   ├── order_items.py          # Order items search tool
│   │   └── order_lookup.py         # Order lookup by status/date tool
│   └── main.py                     # FastAPI app entry point + lifespan
├── data/
│   ├── customers.json              # Seed customer data
│   └── orders.json                 # Seed order data
├── ingestion/
│   ├── faq_knowledge/              # Markdown knowledge base files
│   │   ├── accounts.md
│   │   ├── payments.md
│   │   ├── returns.md
│   │   ├── shipping.md
│   │   └── warranties.md
│   ├── create_checkpoint_tables.py # Postgres checkpoint table setup
│   ├── index_catalog.py            # Product image → CLIP → Qdrant indexer
│   ├── ingest_customers.py         # Seed customers into Supabase
│   ├── ingest_faq.py               # FAQ chunks → embeddings → Qdrant
│   └── ingest_orders.py            # Seed orders into Supabase
├── ui/
│   └── app.py                      # Streamlit chat frontend
├── Dockerfile                      # Multi-stage Docker build
├── start.sh                        # Container entrypoint (FastAPI + Streamlit)
├── pyproject.toml                  # Python dependencies (uv)
├── .env.example                    # Environment variable template
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- **Docker** (optional, for containerized runs)
- API keys for: Groq, Google AI, Nomic, LangSmith
- Accounts on: Supabase, Qdrant Cloud, Redis Cloud

### 1. Clone the Repository

```bash
git clone https://github.com/apurvpanchal-simform/ecomm-cust-assit.git
cd ecomm-cust-assit
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in all required values:

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI API key (for Gemini fallback LLM + embeddings) |
| `GROQ_API_KEY` | Groq API key (for primary LLM inference) |
| `NOMIC_API_KEY` | Nomic API key (for primary text embeddings) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anon/service key |
| `SUPABASE_DB_URL` | Supabase Postgres connection string |
| `REDIS_URL` | Redis connection URL (must support RediSearch) |
| `QDRANT_URL` | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage connection string |
| `AZURE_STORAGE_CONTAINER` | Azure Blob container name (e.g., `product-images`) |
| `JWT_SECRET_KEY` | Secret key for JWT token signing |
| `LANGSMITH_API_KEY` | LangSmith API key for tracing |

### 4. Set Up Local Services (Redis & Qdrant)

If you prefer to run Redis and Qdrant locally instead of using cloud providers, you can instantly spin them up using Docker:

```bash
# Start local Redis
docker run -d -p 6379:6379 -p 8001:8001 --name ecomm-redis redis/redis-stack:latest

# Start local Qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    --name ecomm-qdrant \
    qdrant/qdrant
```

*If running locally, update your `.env`:*
- `REDIS_URL=redis://localhost:6379`
- `QDRANT_URL=http://localhost:6333`
- `QDRANT_API_KEY=` *(leave completely blank)*

### 5. Set Up the Database

Run the SQL schema in your Supabase SQL Editor:

```sql
-- Copy and paste the contents of app/db/schema.sql
```

### 6. Seed Data

```bash
python ingestion/ingest_customers.py    # Seed customers
python ingestion/ingest_orders.py       # Seed orders
python ingestion/ingest_faq.py          # Ingest FAQ knowledge base → Qdrant
python ingestion/index_catalog.py       # Index product images → Qdrant + Azure Blob
```

### 7. Run Locally

```bash
# Terminal 1: Start the FastAPI backend
uvicorn app.main:app --reload --port 8000

# Terminal 2: Start the Streamlit frontend
streamlit run ui/app.py --server.port 8501
```

Open your browser to **http://localhost:8501** to start chatting!

---

## Deployment

### Azure Container Apps (Production)

The project includes a fully automated CI/CD pipeline via GitHub Actions.

#### Prerequisites

1. **Azure Container Registry** — to store Docker images
2. **Azure Container Apps** — serverless container hosting
3. **Azure Blob Storage** — for product images
4. **Cloud Databases** — Supabase (Postgres), Redis Cloud, and Qdrant Cloud must be used (local Docker versions cannot be accessed by Azure Container Apps)

#### GitHub Secrets Required

| Secret | Description |
|---|---|
| `AZURE_CREDENTIALS` | Azure Service Principal credentials JSON |
| `ACR_NAME` | Azure Container Registry name (without `.azurecr.io`) |
| `ACA_APP_NAME` | Azure Container App name |
| `ACA_RESOURCE_GROUP` | Azure Resource Group name |

#### Deploy

Push to `main` or `develop` to trigger an automatic deployment:

```bash
git push origin develop
```

The GitHub Actions workflow will:
1. Check out the repository
2. Authenticate with Azure
3. Build & push the Docker image to ACR
4. Deploy the new image to Azure Container Apps

#### Environment Variables in Azure

Set all `.env` variables as **secrets** in Azure Container Apps → Configuration → Environment Variables.

#### Manage the Container

```bash
# Scale down to zero (stop billing)
az containerapp update --name <app-name> --resource-group <rg> --min-replicas 0 --max-replicas 0

# Scale back up
az containerapp update --name <app-name> --resource-group <rg> --min-replicas 0 --max-replicas 10

# View live logs
az containerapp logs show --name <app-name> --resource-group <rg> --follow
```

---

## API Reference

### Authentication

#### `POST /auth/login`

Authenticate a customer and receive a JWT token.

**Request Body:**
```json
{
  "email": "alice@example.com"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

---

### Chat

All chat endpoints require the `Authorization: Bearer <token>` header.

#### `POST /chat`

Send a message (text and/or image) to the AI assistant.

**Request Body:**
```json
{
  "query": "Where is my latest order?",
  "conversation_id": "optional-thread-id",
  "image_base64": null
}
```

**Response:** Full LangGraph state including `messages` with the AI's response.

#### `GET /chat/conversations`

List all conversations for the authenticated customer.

**Response:**
```json
[
  {
    "conversation_id": "abc123",
    "title": "Where is my latest order...",
    "updated_at": "2026-06-07T15:30:00Z"
  }
]
```

#### `GET /chat/history/{conversation_id}`

Retrieve the full message history for a specific conversation.

**Response:**
```json
{
  "messages": [
    { "role": "user", "content": "Where is my order?" },
    { "role": "assistant", "content": "Your order #ORD-1001 is currently..." }
  ]
}
```

---

### Health Check

#### `GET /health`

Returns the health status of the API.

**Response:**
```json
{
  "status": "ok"
}
```

---

## Data Ingestion

The `ingestion/` directory contains scripts to populate all data stores:

| Script | Target | Description |
|---|---|---|
| `ingest_customers.py` | Supabase | Seeds the `customers` table from `data/customers.json` |
| `ingest_orders.py` | Supabase | Seeds the `orders` table from `data/orders.json` |
| `ingest_faq.py` | Qdrant | Chunks FAQ markdown files, generates embeddings, and upserts into the `ecommerce-knowledge` collection |
| `index_catalog.py` | Qdrant + Azure Blob | Downloads product images, uploads to Azure Blob Storage, generates CLIP embeddings, and indexes into the `product_images` collection |

---

## Observability

This project is fully integrated with **LangSmith** for end-to-end tracing:

- Every agent node is decorated with `@traceable` for automatic span creation
- Token usage and estimated costs are logged per request via `TokenCostCallbackHandler`
- Local token usage logs are written to `data/token_usage.jsonl`

Visit your [LangSmith Dashboard](https://smith.langchain.com/) to view traces, latency breakdowns, and token consumption.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'feat: add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ using LangGraph, FastAPI, and Streamlit
</p>
