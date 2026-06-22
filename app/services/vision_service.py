"""
Azure AI Vision service integration for image analysis and vectorization.
"""

import base64
import logging
import os

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)


def get_vision_client() -> ImageAnalysisClient | None:
    """
    Initializes and returns an Azure ImageAnalysisClient using environment credentials.

    Returns:
        The client instance, or None if credentials are not configured.
    """
    endpoint = os.getenv("AZURE_VISION_ENDPOINT")
    key = os.getenv("AZURE_VISION_KEY")
    if not endpoint or not key:
        logger.warning(
            "Azure Vision credentials missing. Skipping advanced image analysis."
        )
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
        logger.info("\n==============================================")
        logger.info("Hitting Azure AI Vision for Image Analysis...")
        result = client.analyze(
            image_data=image_bytes,
            visual_features=[VisualFeatures.TAGS, VisualFeatures.READ],
        )
        logger.info("Azure AI Vision response received.")

        caption = result.caption.text if result.caption else ""
        tags = [tag.name for tag in result.tags.list] if result.tags else []
        logger.info("-> Extracted Tags: %s", tags)

        ocr_lines = []
        if result.read and result.read.blocks:
            for block in result.read.blocks:
                for line in block.lines:
                    ocr_lines.append(line.text)
        ocr_text = " ".join(ocr_lines)
        logger.info("-> Extracted OCR/Description: %s", ocr_text)
        logger.info("==============================================\n")

        return {"caption": caption, "tags": tags, "ocr_text": ocr_text}
    except Exception as e:
        logger.exception("Azure Vision analysis failed: %s", e)
        return {"caption": "", "tags": [], "ocr_text": ""}


def analyze_image_base64(b64_str: str) -> dict:
    """Analyze a base64-encoded image and return a dictionary of analysis results.

    Args:
        b64_str: Base64-encoded image string.

    Returns:
        dict: Analysis results such as image dimensions, color histogram, and detected objects.
    """
    return analyze_image_bytes(base64.b64decode(b64_str))
