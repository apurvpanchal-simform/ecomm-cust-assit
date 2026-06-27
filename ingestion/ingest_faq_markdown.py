"""
Script to chunk and embed markdown FAQ documents into Qdrant for RAG.
"""

import asyncio
import hashlib
import os
import re
import sys
import uuid
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client import models
from qdrant_client.http.models import PointStruct

# Ensure root directory is in path so we can import 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.retriever import FAQRetriever
from ingestion.utils import setup_database_schema

headers_to_split_on = [
    ("##", "Section"),
    ("###", "Question"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

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
    Read markdown files from the source directory and extract metadata.

    Args:
        source_dir: Path to the directory containing .md files.

    Returns:
        A list of dictionaries containing file content, filename, title, and tags.
    """
    documents = []
    for path in sorted(source_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        
        # Extract Title (first `# ` heading)
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.name
        
        # Extract RAG Tags
        tags_match = re.search(r"^\*\*RAG Tags:\*\*\s+(.+)$", content, re.MULTILINE)
        tags = tags_match.group(1).strip() if tags_match else ""
        
        documents.append(
            {
                "content": content,
                "source_file": path.name,
                "title": title,
                "tags": tags,
            }
        )
    return documents


def _chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk loaded documents using MarkdownHeaderTextSplitter and RecursiveCharacterTextSplitter, inject context, and hash.

    Args:
        documents: List of loaded document dictionaries.

    Returns:
        List of chunk dictionaries with text content and metadata.
    """
    all_chunks: list[dict] = []
    for doc in documents:
        # 1. Semantic Split
        semantic_splits = markdown_splitter.split_text(doc["content"])
        
        # 2. Safety Split
        safety_splits = char_splitter.split_documents(semantic_splits)
        
        for split in safety_splits:
            section = split.metadata.get("Section", "General")
            question = split.metadata.get("Question", "")
            
            # Context Enrichment
            enriched_text = f"Document: {doc['title']}\n"
            if doc["tags"]:
                enriched_text += f"Tags: {doc['tags']}\n"
            enriched_text += f"Section: {section}\n"
            if question:
                enriched_text += f"Question: {question}\n"
            enriched_text += f"---\n{split.page_content.strip()}"
            
            # Chunk Hashing
            chunk_hash = hashlib.md5(enriched_text.encode("utf-8")).hexdigest()
            # Deterministic Point ID based on chunk hash
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc['source_file']}_{chunk_hash}"))

            all_chunks.append(
                {
                    "id": point_id,
                    "text": enriched_text,
                    "source_file": doc["source_file"],
                    "chunk_hash": chunk_hash,
                    "title": doc["title"],
                }
            )
    return all_chunks


async def _sync_chunks_with_qdrant(
    store: FAQRetriever, all_chunks: list[dict]
) -> list[dict]:
    """
    Compare new chunks with Qdrant, delete obsolete ones, and return chunks to embed.

    Args:
        store: FAQRetriever instance containing Qdrant client.
        all_chunks: List of newly generated chunk dictionaries.

    Returns:
        List of chunk dictionaries that are new and need embedding.
    """
    chunks_to_embed = []
    
    # Group chunks by source_file to minimize scroll requests
    chunks_by_file = {}
    for chunk in all_chunks:
        chunks_by_file.setdefault(chunk["source_file"], []).append(chunk)

    for source_file, file_chunks in chunks_by_file.items():
        new_ids = {c["id"] for c in file_chunks}
        
        # Scroll for all existing point IDs for this source_file
        existing_ids = set()
        offset = None
        while True:
            results, offset = await store.client.scroll(
                collection_name=store.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchValue(value=source_file),
                        )
                    ]
                ),
                with_payload=False,
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            existing_ids.update([p.id for p in results])
            if offset is None:
                break
        
        # Diff
        ids_to_delete = existing_ids - new_ids
        ids_to_embed = new_ids - existing_ids
        
        if ids_to_delete:
            print(f"Deleting {len(ids_to_delete)} obsolete chunks from {source_file}")
            await store.client.delete(
                collection_name=store.collection_name,
                points_selector=models.PointIdsList(points=list(ids_to_delete)),
            )
            
        if ids_to_embed:
            print(f"Queueing {len(ids_to_embed)} new chunks from {source_file}")
            for chunk in file_chunks:
                if chunk["id"] in ids_to_embed:
                    chunks_to_embed.append(chunk)
        else:
            print(f"Skipping {source_file} (all chunks identical)")
            
    return chunks_to_embed


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
    chunks_to_embed: list[dict], vectors: list[list[float]]
) -> list[PointStruct]:
    """
    Construct PointStruct objects for Qdrant ingestion using deterministic IDs.

    Args:
        chunks_to_embed: List of chunk metadata dicts.
        vectors: List of corresponding dense vector lists.

    Returns:
        List of PointStruct objects.
    """
    return [
        PointStruct(
            id=chunk["id"],
            vector=vector,
            payload={
                "chunk_id": chunk["id"],
                "content": chunk["text"],
                "source_file": chunk["source_file"],
                "chunk_hash": chunk["chunk_hash"],
                "title": chunk["title"],
            },
        )
        for chunk, vector in zip(chunks_to_embed, vectors)
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
    documents: list[dict], chunks_to_embed: list[dict], points: list[PointStruct]
) -> None:
    """
    Print stats summarizing the document ingestion process.

    Args:
        documents: Initial list of documents.
        chunks_to_embed: List of generated chunks that were embedded.
        points: Final list of points created.
    """
    print(f"Ingested {len(points)} new chunks into Qdrant!")
    print(f"  Files processed      : {len(documents)}")
    print(f"  New Chunks produced  : {len(chunks_to_embed)}")
    if chunks_to_embed:
        avg_size = sum(len(c["text"]) for c in chunks_to_embed) // len(chunks_to_embed)
        print(f"  Avg new chunk size   : {avg_size} chars")


# ── Public Entrypoint ─────────────────────────────────────────────────────────


async def ingest_documents() -> None:
    """
    Reads markdown files from data/faq_knowledge, chunks them using RecursiveCharacterTextSplitter,
    generates dense text embeddings, and upserts them into the Qdrant FAQ collection.
    """
    # ── 1. Apply general app schema ───────────────────────────────────────────
    setup_database_schema()

    # ── 2. Initialize Qdrant collection ───────────────────────────────────────
    store = FAQRetriever(collection_name="ecommerce-knowledge-markdown")
    source_dir = Path("data/faq_knowledge")
    await store.initialize(recreate=False)

    # ── 3. Load and chunk markdown documents ──────────────────────────────────
    documents = _load_markdown_documents(source_dir)
    all_chunks = _chunk_documents(documents)
    
    # ── 4. Sync with Qdrant (Diff & Delete obsolete) ──────────────────────────
    chunks_to_embed = await _sync_chunks_with_qdrant(store, all_chunks)
    
    if not chunks_to_embed:
        print("No new or modified chunks to ingest.")
        return

    # The text is already contextually enriched in _chunk_documents
    texts_to_embed = [c["text"] for c in chunks_to_embed]

    # ── 5. Embed new chunks ───────────────────────────────────────────────────
    vectors = await _embed_chunks(store, texts_to_embed)

    # ── 6. Build points and upsert ────────────────────────────────────────────
    points = _build_qdrant_points(chunks_to_embed, vectors)
    await _upsert_to_qdrant(store, points)

    # ── 7. Print summary report ───────────────────────────────────────────────
    _print_ingestion_summary(documents, chunks_to_embed, points)


if __name__ == "__main__":
    asyncio.run(ingest_documents())
