import os

from dotenv import load_dotenv
from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.services.search import VectorStore
from app.graph.state import AgentState
from app.schemas.response import AgentResponse, TicketCategory

load_dotenv()

SYSTEM_PROMPT = """
You are a customer support agent for an e-commerce platform.

Answer ONLY using the provided context.

Rules:
- Do not invent policies or information.
- If the answer is not in the context, clearly state that.
- Be professional, concise, and helpful.
- Suggest next steps when relevant.
- Extract order_id if present.
- Set requires_human=True when:
  - the context does not answer the question,
  - the user requests a human,
  - the issue requires manual investigation.

CONTEXT:
{context}
"""

class FAQResponse(BaseModel):
    resolution_text: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    requires_human: bool
    suggested_actions: list[str] = Field(default_factory=list)
    escalation_reason: str | None = None
    order_id: str | None = None

@traceable(name="faq_retrieval")
def retrieve_context(
    store: VectorStore,
    query: str,
):
    return store.vector_search(
        query=query,
        top_k=3,
    )

@traceable(name="faq_generation")
def generate_answer(
    llm,
    context: str,
    message: str,
):
    return llm.invoke(
        [
            SystemMessage(
                content=SYSTEM_PROMPT.format(
                    context=context
                )
            ),
            HumanMessage(content=message),
        ]
    )


@traceable(
    name="faq_node",
    metadata={
        "agent": "faq",
        "retrieval_strategy": "cosine",
        "top_k": 3,
    },
)
def faq_node(state: AgentState) -> dict:

    store = VectorStore()

    message = state.get("query", "")

    chunks = retrieve_context(
        store,
        message,
    )

    sources = [
        chunk.get("source_file", "unknown")
        for chunk in chunks
    ]

    context = "\n\n---\n\n".join(
        f"[Source: {chunk.get('source_file', 'unknown')}]\n"
        f"{chunk.get('content', '')}"
        for chunk in chunks
    )

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("PRIMARY_MODEL")
    )

    structured_llm = llm.with_structured_output(
        FAQResponse
    )

    faq_response = generate_answer(
        structured_llm,
        context,
        message,
    )

    agent_response = AgentResponse(
        resolution_text=faq_response.resolution_text,
        confidence_score=faq_response.confidence_score,
        ticket_category=TicketCategory.FAQ,
        requires_human=faq_response.requires_human,
        sources=list(set(sources)),
        suggested_actions=faq_response.suggested_actions,
        escalation_reason=faq_response.escalation_reason,
        order_id=faq_response.order_id,
    )

    return {
        **state,
        "agent_response": agent_response,
        "structured_output": agent_response.model_dump(
            mode="json"
        ),
    }