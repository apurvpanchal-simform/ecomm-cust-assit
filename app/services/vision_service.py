import os
import logging
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import base64
import requests

logger = logging.getLogger(__name__)


def get_vision_client() -> ImageAnalysisClient | None:
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
        logger.info(f"-> Extracted Tags: {tags}")

        ocr_lines = []
        if result.read and result.read.blocks:
            for block in result.read.blocks:
                for line in block.lines:
                    ocr_lines.append(line.text)
        ocr_text = " ".join(ocr_lines)
        logger.info(f"-> Extracted OCR/Description: {ocr_text}")
        logger.info("==============================================\n")

        return {"caption": caption, "tags": tags, "ocr_text": ocr_text}
    except Exception as e:
        logger.exception(f"Azure Vision analysis failed: {e}")
        return {"caption": "", "tags": [], "ocr_text": ""}


def analyze_image_base64(b64_str: str) -> dict:
    return analyze_image_bytes(base64.b64decode(b64_str))





def vectorize_image_bytes(image_bytes: bytes) -> list[float]:
    """Generates a 1024d multimodal embedding using Azure AI Vision (Florence)."""
    endpoint = os.getenv("AZURE_VISION_ENDPOINT")
    key = os.getenv("AZURE_VISION_KEY")
    if not endpoint or not key:
        logger.warning("Azure Vision credentials missing for embeddings.")
        return []

    # Ensure endpoint ends with a slash
    if not endpoint.endswith("/"):
        endpoint += "/"

    url = f"{endpoint}computervision/retrieval:vectorizeImage?api-version=2024-02-01&model-version=2023-04-15"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/octet-stream",
    }

    try:
        logger.info("Hitting Azure AI Vision for Image Embedding...")
        response = requests.post(url, headers=headers, data=image_bytes)
        response.raise_for_status()
        return response.json().get("vector", [])
    except Exception as e:
        logger.exception(f"Azure Vision vectorizeImage failed: {e}")
        return []


def vectorize_image_base64(b64_str: str) -> list[float]:
    return vectorize_image_bytes(base64.b64decode(b64_str))


def vectorize_text(text: str) -> list[float]:
    """Generates a 1024d multimodal embedding using Azure AI Vision (Florence)."""
    endpoint = os.getenv("AZURE_VISION_ENDPOINT")
    key = os.getenv("AZURE_VISION_KEY")
    if not endpoint or not key:
        return []

    if not endpoint.endswith("/"):
        endpoint += "/"

    url = f"{endpoint}computervision/retrieval:vectorizeText?api-version=2024-02-01&model-version=2023-04-15"
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"}

    try:
        logger.info(f"Hitting Azure AI Vision for Text Embedding: '{text}'")
        response = requests.post(url, headers=headers, json={"text": text})
        response.raise_for_status()
        return response.json().get("vector", [])
    except Exception as e:
        logger.exception(f"Azure Vision vectorizeText failed: {e}")
        return []
