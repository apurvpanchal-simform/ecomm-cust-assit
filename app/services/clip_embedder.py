import os
import torch
from PIL import Image
import io, base64
from functools import lru_cache
from transformers import AutoProcessor, AutoModel


@lru_cache(maxsize=1)
def load_model():
    model_name = os.getenv("CLIP_MODEL", "google/siglip-base-patch16-224")
    hf_token = os.getenv("HF_TOKEN")
    try:
        processor = AutoProcessor.from_pretrained(
            model_name, token=hf_token, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            model_name, token=hf_token, local_files_only=True
        )
    except Exception:
        import logging

        logging.getLogger(__name__).info(
            "Model not found in cache, downloading from HuggingFace..."
        )
        processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
        model = AutoModel.from_pretrained(model_name, token=hf_token)
    model.eval()
    return model, processor


def embed_image_bytes(image_bytes: bytes) -> list[float]:
    model, processor = load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features.pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze().tolist()


def embed_image_base64(b64_str: str) -> list[float]:
    return embed_image_bytes(base64.b64decode(b64_str))


def embed_text(text: str) -> list[float]:
    """Embed text query into same SigLIP2 space — enables text → image search."""
    model, processor = load_model()
    inputs = processor(text=[text], padding="max_length", return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.squeeze().tolist()
