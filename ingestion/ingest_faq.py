import os
import asyncio
import sys
import uuid
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import PointStruct

# Ensure root directory is in path so we can import 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.search import VectorStore


async def ingest_documents():
    store = VectorStore()
    source_dir = Path("ingestion/faq_knowledge")
    await store.initialize()

    documents = []
    for path in source_dir.glob("*.md"):
        content = path.read_text(encoding="utf-8")
        documents.append({"content": content, "source_file": path.name})

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    points = []

    for doc in documents:
        chunks = splitter.split_text(doc["content"])
        for chunk_text in chunks:
            vector = store.embeddings.embed_query(chunk_text)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"content": chunk_text, "source_file": doc["source_file"]},
                )
            )

    if points:
        await store.client.upsert(collection_name=store.collection_name, points=points)
        print(f"Ingested {len(points)} chunks into Qdrant !")


if __name__ == "__main__":
    asyncio.run(ingest_documents())
