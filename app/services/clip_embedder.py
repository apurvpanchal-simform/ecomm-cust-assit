import os
import open_clip
import torch
from PIL import Image
import io, base64
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model():
    clip_model = os.getenv("CLIP_MODEL", "ViT-B-32")
    clip_pretrained = os.getenv("CLIP_PRETRAINED", "openai")
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(clip_model)
    return model, preprocess, tokenizer

def embed_image_bytes(image_bytes: bytes) -> list[float]:
    model, preprocess, _ = load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        vector = model.encode_image(tensor)
        vector = vector / vector.norm(dim=-1, keepdim=True)   # L2 normalize
    return vector.squeeze().tolist()

def embed_image_base64(b64_str: str) -> list[float]:
    return embed_image_bytes(base64.b64decode(b64_str))

def embed_text(text: str) -> list[float]:
    """Embed text query into same CLIP space — enables text → image search."""
    model, _, tokenizer = load_model()
    tokens = tokenizer([text])
    with torch.no_grad():
        vector = model.encode_text(tokens)
        vector = vector / vector.norm(dim=-1, keepdim=True)
    return vector.squeeze().tolist()
