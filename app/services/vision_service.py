"""
Azure AI Vision service integration for image analysis and content safety.

Exposes two public helpers:
- `analyze_image_bytes(image_bytes)` — analyze raw image bytes
- `analyze_image_base64(b64_str)`   — analyze a base64-encoded image

Both return a dict with keys:
  caption  (str)       — a short natural-language description of the image
  tags     (list[str]) — object/concept tags detected by Azure CV
  ocr_text (str)       — any text visible in the image (OCR)

The Azure client is lazily created on each call and gracefully returns empty
results if credentials are not configured (e.g., in local dev environments).
"""

import base64
import logging
import os

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)


# ── Client factory ────────────────────────────────────────────────────────────


def get_vision_client() -> ImageAnalysisClient | None:
    """
    Initialise and return an Azure ImageAnalysisClient using environment credentials.

    Reads AZURE_VISION_ENDPOINT and AZURE_VISION_KEY from the environment.

    Returns:
        A configured ImageAnalysisClient, or None if the credentials are missing.
    """
    endpoint = os.getenv("AZURE_VISION_ENDPOINT")
    key = os.getenv("AZURE_VISION_KEY")

    if not endpoint or not key:
        logger.warning(
            "Azure Vision credentials missing. Skipping advanced image analysis."
        )
        return None

    return ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# ── Analysis functions ────────────────────────────────────────────────────────


def analyze_image_bytes(image_bytes: bytes) -> dict:
    """
    Analyse raw image bytes using Azure AI Vision.

    Requests TAGS and READ (OCR) features from the Azure API. Caption analysis
    is included if the service returns it, but is not explicitly requested because
    it requires a different API tier.

    Args:
        image_bytes: Raw bytes of the image (JPEG, PNG, etc.).

    Returns:
        A dict with:
            caption  (str)       — natural-language image description, or ""
            tags     (list[str]) — detected object/concept tags, or []
            ocr_text (str)       — space-joined OCR lines from the image, or ""
        Returns all-empty values if the Azure client is unavailable or the call fails.
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

        # ── Extract caption (may be None if unsupported by the tier) ─────────
        caption = result.caption.text if result.caption else ""

        # ── Extract tags as plain strings ─────────────────────────────────────
        tags = [tag.name for tag in result.tags.list] if result.tags else []
        logger.info("-> Extracted Tags: %s", tags)

        # ── Extract OCR text from all read blocks and lines ───────────────────
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
    """
    Analyse a base64-encoded image using Azure AI Vision.

    A thin convenience wrapper around `analyze_image_bytes` that handles the
    base64 decoding step.

    Args:
        b64_str: Standard base64-encoded image string (no data-URL prefix needed).

    Returns:
        Same dict as `analyze_image_bytes`: {caption, tags, ocr_text}.
    """
    return analyze_image_bytes(base64.b64decode(b64_str))
