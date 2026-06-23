"""
Pre-processing graph node that analyses user-uploaded images before the supervisor runs.

`image_analyzer_node` is the first node to execute when the user includes an image
in their message.  It calls Azure AI Vision to extract a caption, object tags, and
any visible text (OCR), then checks the tags against a safety blocklist.

If the image is safe, the extracted metadata is stored in the state so the supervisor
and image_search_node can use it for routing and product matching.

If the image is unsafe (e.g. contains weapons or NSFW content), the raw image data
is removed from the state and a safety warning is injected so downstream agents can
politely decline the image while still processing any valid text in the same message.
"""

import asyncio
import logging

from langfuse import observe

from app.graph.state import AgentState
from app.services.vision_service import analyze_image_base64

logger = logging.getLogger(__name__)

# Tags that trigger an immediate safety block.
# Any overlap with Azure CV tags causes the image to be rejected.
_UNSAFE_TAGS = {
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


# ── Private helpers ───────────────────────────────────────────────────────────


def _build_image_description(caption: str, ocr_text: str) -> str | None:
    """
    Build a single human-readable description string from caption and OCR text.

    Both parts are optional — only non-empty values are included.  The parts are
    joined with " | " to clearly separate them.

    Args:
        caption: Natural-language image description from Azure CV (may be "").
        ocr_text: Any visible text extracted via OCR (may be "").

    Returns:
        A formatted description string, or None if both inputs are empty.
    """
    parts = []
    if caption:
        parts.append(f"Caption: {caption}")
    if ocr_text:
        parts.append(f"Visible Text/Brand: {ocr_text}")
    return " | ".join(parts) if parts else None


def _check_image_safety(tags: list[str]) -> bool:
    """
    Check whether any of the detected tags are on the safety blocklist.

    Args:
        tags: List of tag name strings returned by Azure CV.

    Returns:
        True if the image should be blocked, False if it is safe.
    """
    return any(tag.lower() in _UNSAFE_TAGS for tag in tags)


def _build_unsafe_state_update() -> dict:
    """
    Build the state update dict returned when an unsafe image is detected.

    Clears the image data so the unsafe image does not persist into subsequent
    nodes, and injects a safety warning for downstream agents.

    Returns:
        A partial AgentState dict with image data nulled and a warning injected.
    """
    return {
        "image_base64": None,
        "image_is_safe": False,
        "image_tags": [],
        "image_description": None,
        "image_safety_warning": (
            "The user's uploaded image was blocked and removed due to safety violations "
            "(e.g., weapons or NSFW). Please inform the user politely that their image was "
            "rejected, but DO STILL process any valid text requests they made in the same query."
        ),
    }


# ── Node ──────────────────────────────────────────────────────────────────────


@observe(name="image_analyzer_node")
async def image_analyzer_node(state: AgentState) -> dict:
    """
    Analyse an uploaded image and update the state with extracted metadata or a safety block.

    Flow:
    1. Check if `image_base64` is present in the state.  If not, return {} (no-op).
    2. Call Azure AI Vision to extract caption, tags, and OCR text.
    3. Build a combined image description string from the caption and OCR output.
    4. Check extracted tags against `_UNSAFE_TAGS`.
    5a. If unsafe: null out image data and inject a safety warning into the state.
    5b. If safe: store tags, description, and `image_is_safe=True` in the state.

    Args:
        state: The global AgentState containing `image_base64` (or not).

    Returns:
        A partial state dict to merge.  Returns {} if no image is present.
        On success, includes `image_tags`, `image_description`, `image_is_safe`.
        On safety block, includes nulled image fields and `image_safety_warning`.
        On unexpected error, returns {} so the graph continues without crashing.
    """
    image_base64 = state.get("image_base64")

    # ── 1. No image present — pass through without doing anything ────────────
    if not image_base64:
        return {}

    try:
        # ── 2. Analyse image with Azure AI Vision (run in thread — blocking I/O) ─
        analysis = await asyncio.to_thread(analyze_image_base64, image_base64)

        caption = analysis.get("caption", "")
        ocr_text = analysis.get("ocr_text", "")
        tags = analysis.get("tags", [])

        # ── 3. Build combined description for downstream nodes ────────────────
        image_description = _build_image_description(caption, ocr_text)

        logger.info("Azure CV Extracted Tags: %s", tags)
        logger.info("Azure CV Extracted OCR/Caption: %s", image_description)

        # ── 4. Safety check — block if any unsafe tag is detected ─────────────
        if _check_image_safety(tags):
            logger.warning("Safety violation blocked image with tags: %s", tags)
            return _build_unsafe_state_update()

        # ── 5. Safe image — store metadata for supervisor and image_search ─────
        return {
            "image_tags": tags,
            "image_description": image_description,
            "image_safety_warning": None,  # Explicitly clear any previous warning
            "image_is_safe": True,
        }

    except Exception as e:
        logger.exception("Image analyzer failed: %s", e)
        # Return empty dict — do not crash the graph; supervisor will run without image context
        return {}
