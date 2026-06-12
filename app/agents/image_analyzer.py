from typing import Any
from langfuse import observe
from app.graph.state import AgentState
from app.services.vision_service import analyze_image_base64
from langchain_core.messages import SystemMessage
import logging

logger = logging.getLogger(__name__)


@observe(name="image_analyzer_node")
async def image_analyzer_node(state: AgentState) -> dict:
    """
    Pre-processes user-uploaded images using Azure AI Vision before they reach the supervisor.

    This node acts as a middleware step at the very beginning of the LangGraph execution. 
    If an image is detected in the global state, this node extracts descriptive features 
    (tags and OCR text) to provide textual context for downstream agents. It also implements 
    safety mechanisms to block NSFW or dangerous content.

    Flow:
    1. Checks if `image_base64` is present in the state. If not, bypasses execution.
    2. Calls the Azure Vision service to analyze the image.
    3. Extracts generated tags and OCR text (if any) and formats them into a descriptive string.
    4. Evaluates the extracted tags against a predefined list of `UNSAFE_TAGS`.
    5. If an unsafe tag is found, it nullifies the image data and injects a SystemMessage 
       warning downstream agents about the safety violation, prompting them to politely decline 
       the image while answering the textual part of the query.
    6. If the image is safe, it updates the state with `image_tags` and `image_description`.

    Args:
        state (AgentState): The global state of the conversation containing `image_base64`.

    Returns:
        dict: A dictionary containing the `image_tags` and `image_description` to merge into 
              the state. If a safety violation occurs, returns empty image data and a warning message.
    """

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
