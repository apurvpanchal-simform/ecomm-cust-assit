from langchain_core.tools import tool
from app.services.search import VectorStore
from langsmith import traceable

@tool
@traceable(name="tool_search_faq")
def search_faq(query: str) -> str:
    """Search the company knowledge base for policies, shipping, returns, and general info.
    Use this tool whenever the user asks a general question about the company.
    """
    store = VectorStore()
    chunks = store.vector_search(query=query, top_k=3)
    
    if not chunks:
        return "No relevant FAQ articles found."

    context = "\n\n---\n\n".join(
        f"[Source: {chunk.get('source_file', 'unknown')}]\n"
        f"{chunk.get('content', '')}"
        for chunk in chunks
    )
    return context