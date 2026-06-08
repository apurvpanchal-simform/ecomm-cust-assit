import os
import logging
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)

def get_vision_client() -> ImageAnalysisClient | None:
    endpoint = os.getenv("AZURE_VISION_ENDPOINT")
    key = os.getenv("AZURE_VISION_KEY")
    if not endpoint or not key:
        logger.warning("Azure Vision credentials missing. Skipping advanced image analysis.")
        return None
    
    return ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def analyze_image_bytes(image_bytes: bytes) -> dict:
    """
    Analyzes an image and extracts a caption, tags, and OCR text.
    Returns a dict with 'caption', 'tags', and 'ocr_text'.
    """
    client = get_vision_client()
    if not client:
        return {"caption": "", "tags": [], "ocr_text": ""}
        
    try:
        result = client.analyze(
            image_data=image_bytes,
            visual_features=[
                VisualFeatures.TAGS,
                VisualFeatures.READ
            ]
        )
        
        # We removed CAPTION because it is unsupported in some Azure regions
        caption = ""
        tags = [tag.name for tag in result.tags.list] if result.tags else []
        
        ocr_lines = []
        if result.read and result.read.blocks:
            for block in result.read.blocks:
                for line in block.lines:
                    ocr_lines.append(line.text)
        ocr_text = " ".join(ocr_lines)
        
        return {
            "caption": caption,
            "tags": tags,
            "ocr_text": ocr_text
        }
    except Exception as e:
        logger.exception(f"Azure Vision analysis failed: {e}")
        return {"caption": "", "tags": [], "ocr_text": ""}

def analyze_image_base64(b64_str: str) -> dict:
    import base64
    return analyze_image_bytes(base64.b64decode(b64_str))
