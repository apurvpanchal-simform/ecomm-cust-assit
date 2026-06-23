"""
Tool for searching the FAQ knowledge base via semantic (vector) search.

Wraps FAQRetriever and applies a confidence-score threshold before
returning context chunks to the FAQ agent.
"""

import logging

from langchain_core.tools import tool
from langfuse import observe

from app.config import SEARCH_CONFIG
from app.rag.retriever import FAQRetriever
from app.schemas.agent import ToolNotFoundResponse

logger = logging.getLogger(__name__)


@tool
@observe(name="tool_search_faq")
async def search_faq(query: str) -> dict:
    """
    Search the company knowledge base for policies, shipping, returns, and general info.

    Use this tool whenever the user asks a general question about the company.

    Args:
        query: The user's specific question or keywords to search for.

    Returns:
        A dictionary containing the aggregated context string and individual chunks.
    """
    store = FAQRetriever()
    logger.info("🔍 FAQ Search Triggered! Query: '%s'", query)

    # ── 1. Run vector search against Qdrant ───────────────────────────────────
    chunks = await store.vector_search(query=query, top_k=SEARCH_CONFIG.faq_top_k)

    # ── 2. Log raw results for observability ──────────────────────────────────
    log_msg = "\n========== RAW QDRANT FAQ RESULTS ==========\n"
    if not chunks:
        log_msg += "No chunks found.\n"
    for idx, c in enumerate(chunks, 1):
        log_msg += (
            f"{idx}. [Score: {c.get('score', 0):.4f}] "
            f"Source: {c.get('source_file')} -> {c.get('content', '')[:100]}...\n"
        )
    log_msg += "============================================\n"
    logger.info(log_msg)

    # ── 3. Return early if nothing was found ──────────────────────────────────
    if not chunks:
        response = ToolNotFoundResponse(
            message="No relevant FAQ articles found in the knowledge base."
        )
        return {"context": response.model_dump(mode="json"), "chunks": []}

    # ── 4. Apply confidence threshold ─────────────────────────────────────────
    # If the best result's similarity is below the threshold, the search is too uncertain
    # to give a reliable answer — return a polite fallback instead.
    best_score = max(chunk.get("score", 0.0) for chunk in chunks)
    if best_score < SEARCH_CONFIG.faq_search_threshold:
        response = ToolNotFoundResponse(
            message=(
                "I'm sorry, but I don't have a confident answer to this question "
                "based on our knowledge base. Please try rephrasing your query "
                "or contact our human support team for further assistance."
            )
        )
        return {"context": response.model_dump(mode="json"), "chunks": []}

    # ── 5. Filter and aggregate confident chunks into a single context block ──
    confident_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("score", 0.0) >= SEARCH_CONFIG.faq_search_threshold
    ]

    context = "\n\n---\n\n".join(
        f"{chunk.get('content', '')}" for chunk in confident_chunks
    )
    return {"context": context, "chunks": confident_chunks}
