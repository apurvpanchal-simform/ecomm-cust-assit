from functools import lru_cache
from typing import Any, Dict

from fastembed import SparseTextEmbedding


@lru_cache(maxsize=1)
def load_sparse_model():
    """Loads and returns a pre-trained sparse model from the default storage location. The returned model is a scipy.sparse matrix suitable for inference."""
    # Load the highly optimized Qdrant BM25 model
    model = SparseTextEmbedding("Qdrant/bm25")
    return model


def embed_sparse_text(text: str) -> Dict[str, Any]:
    """
    Generate BM25 sparse vectors for the given text.
    Returns a dictionary with 'indices' and 'values' needed by Qdrant.
    """
    if not text or not text.strip():
        return {"indices": [], "values": []}

    model = load_sparse_model()
    # fastembed returns an iterator of SparseEmbedding objects
    # We only pass one text, so we take the first element
    result = list(model.embed([text]))[0]

    return {"indices": result.indices.tolist(), "values": result.values.tolist()}
