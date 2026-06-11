from typing import Any
from langsmith import traceable
from app.graph.state import AgentState
from app.services.azure_vision import analyze_image_base64
from langchain_core.messages import SystemMessage
import logging

logger = logging.getLogger(__name__)


@traceable(name="image_analyzer_node")
async def image_analyzer_node(state: AgentState) -> dict:
    """Pre-processes images using Azure Vision OCR before the supervisor runs."""

    image_base64 = state.get("image_base64")

    # If no image, pass through cleanly
    if not image_base64:
        return {}

    try:
        analysis = analyze_image_base64(image_base64)

        caption = analysis.get("caption", "")
        ocr_text = analysis.get("ocr_text", "")
        tags = analysis.get("tags", [])

        description_parts = []
        if caption:
            description_parts.append(f"Caption: {caption}")
        if ocr_text:
            description_parts.append(f"Visible Text/Brand: {ocr_text}")

        image_description = " | ".join(description_parts) if description_parts else None

        logger.info(f"Azure CV Extracted Tags: {tags}")
        logger.info(f"Azure CV Extracted OCR/Caption: {image_description}")

        UNSAFE_TAGS = {
            "weapon",
            "gun",
            "firearm",
            "rifle",
            "pistol",
            "revolver",
            "nude",
            "nsfw",
            "violence",
            "blood",
            "gore",
            "explosive",
        }
        unsafe_found = any(tag.lower() in UNSAFE_TAGS for tag in tags)

        if unsafe_found:
            logger.warning(f"Safety violation blocked image with tags: {tags}")
            return {
                "image_base64": None,
                "image_tags": [],
                "image_description": None,
                "messages": [
                    SystemMessage(
                        content="SYSTEM: The user's uploaded image was blocked and removed due to safety violations (e.g., weapons or NSFW). Please inform the user politely that their image was rejected, but DO STILL process any valid text requests they made in the same query."
                    )
                ],
            }

        return {"image_tags": tags, "image_description": image_description}
    except Exception as e:
        logger.exception(f"Image analyzer failed: {e}")
        return {}
