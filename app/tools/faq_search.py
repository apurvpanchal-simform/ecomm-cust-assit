from langchain_core.tools import tool
from app.services.search import VectorStore
from app.schemas.agent import ToolNotFoundResponse
from langsmith import traceable


@tool
@traceable(name="tool_search_faq")
async def search_faq(query: str) -> str:
    """Search the company knowledge base for policies, shipping, returns, and general info.
    Use this tool whenever the user asks a general question about the company.
    """
    store = VectorStore()
    chunks = await store.vector_search(query=query, top_k=3)

    if not chunks:
        response = ToolNotFoundResponse(
            message="No relevant FAQ articles found in the knowledge base."
        )
        return response.model_dump(mode="json")

    # Check if the best result's confidence score meets the threshold
    best_score = max(chunk.get("score", 0.0) for chunk in chunks)
    if best_score < 0.7:
        response = ToolNotFoundResponse(
            message=(
                "I'm sorry, but I don't have a confident answer to this question "
                "based on our knowledge base. Please try rephrasing your query "
                "or contact our human support team for further assistance."
            )
        )
        return response.model_dump(mode="json")

    # Only include chunks that meet the confidence threshold
    confident_chunks = [chunk for chunk in chunks if chunk.get("score", 0.0) >= 0.7]

    context = "\n\n---\n\n".join(
        f"[Source: {chunk.get('source_file', 'unknown')}]\n"
        f"{chunk.get('content', '')}"
        for chunk in confident_chunks
    )
    return context

