"""
Script to chunk and embed markdown FAQ documents into Qdrant for RAG.
"""

import asyncio
import os
import sys
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import PointStruct

# Ensure root directory is in path so we can import 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.retriever import FAQRetriever
from ingestion.utils import setup_database_schema

# ── Chunking config ────────────────────────────────────────────────────────────
# Industry standard ideal chunk size and overlap for dense embeddings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", " ", ""]

char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
)


# ── Private helpers ───────────────────────────────────────────────────────────


def _load_markdown_documents(source_dir: Path) -> list[dict]:
    """
    Read markdown files from the source directory.

    Args:
        source_dir: Path to the directory containing .md files.

    Returns:
        A list of dictionaries containing file content and filename.
    """
    documents = []
    for path in sorted(source_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        documents.append({"content": content, "source_file": path.name})
    return documents


def _chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk loaded documents using RecursiveCharacterTextSplitter.

    Args:
        documents: List of loaded document dictionaries.

    Returns:
        List of chunk dictionaries with text content and metadata.
    """
    all_chunks: list[dict] = []
    for doc in documents:
        sub_texts = char_splitter.split_text(doc["content"])
        for sub_text in sub_texts:
            all_chunks.append(
                {
                    "text": sub_text.strip(),
                    "source_file": doc["source_file"],
                }
            )
    return all_chunks


async def _embed_chunks(
    store: FAQRetriever, texts_to_embed: list[str]
) -> list[list[float]]:
    """
    Embed texts in batches, introducing rate-limiting sleeps if necessary.

    Args:
        store: FAQRetriever instance for accessing embeddings.
        texts_to_embed: List of chunk text strings.

    Returns:
        List of embedded dense vectors.
    """
    vectors = []
    batch_size = 50
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i : i + batch_size]
        print(
            f"Embedding batch {i // batch_size + 1}/{(len(texts_to_embed) - 1) // batch_size + 1}..."
        )
        batch_vectors = await store.embeddings.aembed_documents(batch)
        vectors.extend(batch_vectors)
        if i + batch_size < len(texts_to_embed):
            print("Waiting 60 seconds to respect rate limits...")
            await asyncio.sleep(60)
    return vectors


def _build_qdrant_points(
    all_chunks: list[dict], vectors: list[list[float]]
) -> list[PointStruct]:
    """
    Construct PointStruct objects for Qdrant ingestion.

    Args:
        all_chunks: List of chunk metadata dicts.
        vectors: List of corresponding dense vector lists.

    Returns:
        List of PointStruct objects.
    """
    return [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "content": chunk["text"],
                "source_file": chunk["source_file"],
            },
        )
        for chunk, vector in zip(all_chunks, vectors)
    ]


async def _upsert_to_qdrant(store: FAQRetriever, points: list[PointStruct]) -> None:
    """
    Upsert points into Qdrant collection.

    Args:
        store: FAQRetriever instance containing Qdrant client.
        points: List of points to upsert.
    """
    await store.client.upsert(
        collection_name=store.collection_name,
        points=points,
    )


def _print_ingestion_summary(
    documents: list[dict], all_chunks: list[dict], points: list[PointStruct]
) -> None:
    """
    Print stats summarizing the document ingestion process.

    Args:
        documents: Initial list of documents.
        all_chunks: List of generated chunks.
        points: Final list of points created.
    """
    print(f"Ingested {len(points)} chunks into Qdrant!")
    print(f"  Files processed : {len(documents)}")
    print(f"  Chunks produced : {len(all_chunks)}")
    if all_chunks:
        avg_size = sum(len(c["text"]) for c in all_chunks) // len(all_chunks)
        print(f"  Avg chunk size  : {avg_size} chars")


# ── Public Entrypoint ─────────────────────────────────────────────────────────


async def ingest_documents() -> None:
    """
    Reads markdown files from data/faq_knowledge, chunks them using RecursiveCharacterTextSplitter,
    generates dense text embeddings, and upserts them into the Qdrant FAQ collection.
    """
    # ── 1. Apply general app schema ───────────────────────────────────────────
    setup_database_schema()

    # ── 2. Initialize Qdrant collection ───────────────────────────────────────
    store = FAQRetriever()
    source_dir = Path("data/faq_knowledge")
    await store.initialize(recreate=True)

    # ── 3. Load and chunk markdown documents ──────────────────────────────────
    documents = _load_markdown_documents(source_dir)
    all_chunks = _chunk_documents(documents)

    if not all_chunks:
        print("No chunks produced — check that data/faq_knowledge/ contains .md files.")
        return

    # Prep text payload with Source Context Injection
    texts_to_embed = [
        f"Source Document: {c['source_file']}\n\n{c['text']}" for c in all_chunks
    ]

    # ── 4. Embed chunks ───────────────────────────────────────────────────────
    vectors = await _embed_chunks(store, texts_to_embed)

    # ── 5. Build points and upsert ────────────────────────────────────────────
    points = _build_qdrant_points(all_chunks, vectors)
    await _upsert_to_qdrant(store, points)

    # ── 6. Print summary report ───────────────────────────────────────────────
    _print_ingestion_summary(documents, all_chunks, points)


if __name__ == "__main__":
    asyncio.run(ingest_documents())
