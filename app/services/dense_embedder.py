"""
Dense (SigLIP2) image and text embedding utilities.

Uses the HuggingFace `google/siglip-base-patch16-224` model (or a value
overridden via the CLIP_MODEL environment variable) to produce L2-normalised
384-dimensional vectors in a shared visual-semantic space so that text queries
can be matched directly against image embeddings.

Model loading is cached with `@lru_cache` so it only happens once per process.
"""

import base64
import io
import logging
import os
from functools import lru_cache

import torch

from app.config import SEARCH_CONFIG
from PIL import Image
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


# ── Model loading ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def load_clip_model():
    """
    Load the SigLIP2 vision-language model and its processor.

    Tries to load from the local HuggingFace cache first (fast, offline-safe).
    Falls back to downloading from HuggingFace Hub if the model is not cached.

    The model is set to `eval()` mode and cached so subsequent calls return the
    same objects without re-loading weights.

    Returns:
        A (model, processor) tuple ready for inference.
    """
    model_name = SEARCH_CONFIG.clip_model
    hf_token = SEARCH_CONFIG.hf_token

    try:
        # Prefer the local cache to avoid network I/O in production
        processor = AutoProcessor.from_pretrained(
            model_name, token=hf_token, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            model_name, token=hf_token, local_files_only=True
        )
    except Exception:
        logger.info("Model not found in cache, downloading from HuggingFace...")
        processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
        model = AutoModel.from_pretrained(model_name, token=hf_token)

    # Inference mode — disable gradient tracking for speed and memory efficiency
    model.eval()
    return model, processor


# ── Embedding functions ───────────────────────────────────────────────────────


def embed_image_bytes(image_bytes: bytes) -> list[float]:
    """
    Generate a dense vector embedding from raw image bytes.

    Decodes the bytes into a PIL image, passes it through the SigLIP2 image
    encoder, and returns the L2-normalised pooler output as a Python list.

    Args:
        image_bytes: Raw bytes of the image (JPEG, PNG, etc.).

    Returns:
        A list of floats representing the normalised image embedding vector.
    """
    model, processor = load_clip_model()

    # Convert raw bytes → PIL RGB image → model-ready tensors
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        # Use pooler output for a single fixed-size representation
        image_features = image_features.pooler_output
        # L2-normalise so cosine similarity == dot product
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.squeeze().tolist()


def embed_image_base64(b64_str: str) -> list[float]:
    """
    Generate a dense vector embedding from a base64-encoded image string.

    A thin convenience wrapper around `embed_image_bytes` that handles the
    base64 decoding step.

    Args:
        b64_str: Standard base64-encoded image string (no data-URL prefix needed).

    Returns:
        A list of floats representing the normalised image embedding vector.
    """
    return embed_image_bytes(base64.b64decode(b64_str))


def embed_text(text: str) -> list[float]:
    """
    Generate a dense vector embedding from a text query.

    Projects the text into the *same* visual-semantic space used by
    `embed_image_bytes`, enabling text → image similarity search.

    Args:
        text: The search query string (e.g. "red running shoes under $50").

    Returns:
        A list of floats representing the normalised text embedding vector.
    """
    model, processor = load_clip_model()

    # Pad to max_length so the tensor shape is consistent regardless of query length
    inputs = processor(text=[text], padding="max_length", return_tensors="pt")

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features.pooler_output
        # L2-normalise — same as image embeddings so dot-product == cosine similarity
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.squeeze().tolist()
