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

# ── Chunking config ────────────────────────────────────────────────────────────
# Industry standard ideal chunk size and overlap for dense embeddings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# We use the recommended markdown separators to respect paragraph and structural boundaries.
SEPARATORS = ["\n\n", "\n", " ", ""]
# ──────────────────────────────────────────────────────────────────────────────

char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
)


async def ingest_documents():
    """
    Reads markdown files from data/faq_knowledge, chunks them using RecursiveCharacterTextSplitter,
    generates dense text embeddings, and upserts them into the Qdrant FAQ collection.
    """
    from ingestion.utils import setup_database_schema

    setup_database_schema()

    store = FAQRetriever()
    source_dir = Path("data/faq_knowledge")
    await store.initialize(recreate=True)

    # ── 1. Load markdown files ─────────────────────────────────────────────────
    documents = []
    for path in sorted(source_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        documents.append({"content": content, "source_file": path.name})

    # ── 2. Chunk using RecursiveCharacterTextSplitter ──────────────────────────
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

    if not all_chunks:
        print("No chunks produced — check that data/faq_knowledge/ contains .md files.")
        return

    texts_to_embed = [
        f"Source Document: {c['source_file']}\n\n{c['text']}" for c in all_chunks
    ]

    # ── 3. Batch-embed all chunks with Source Context Injection ────────────────
    # We prepend the filename (e.g., 'returns.md') to help the dense model maintain
    # the overarching context of the chunk.
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

    # ── 4. Build Qdrant points with metadata payload ──────────────────────────
    points = [
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

    # ── 5. Upsert into Qdrant ─────────────────────────────────────────────────
    await store.client.upsert(
        collection_name=store.collection_name,
        points=points,
    )

    print(f"Ingested {len(points)} chunks into Qdrant !")
    print(f"  Files processed : {len(documents)}")
    print(f"  Chunks produced : {len(all_chunks)}")
    print(
        f"  Avg chunk size  : {sum(len(c['text']) for c in all_chunks) // len(all_chunks)} chars"
    )


if __name__ == "__main__":
    asyncio.run(ingest_documents())
