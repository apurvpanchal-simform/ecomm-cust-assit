import os
import base64
from typing import Any
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from app.services.azure_blob import upload_user_image

def get_cv_client():
    endpoint = os.environ.get("AZURE_VISION_ENDPOINT", "")
    key = os.environ.get("AZURE_VISION_KEY", "")
    if not endpoint or not key or key == "<your_key>":
        return None
    
    return ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

def image_analyzer_node(state: Any) -> dict:
    """Use Azure Computer Vision to extract semantic tags from user image."""
    # Assuming state is a dict-like AgentState
    if not state.get("image_is_safe", True) or not state.get("image_base64"):
        return state

    img_bytes = base64.b64decode(state["image_base64"])
    
    cv_client = get_cv_client()
    if not cv_client:
        # Graceful fallback if no API key is provided
        try:
            azure_url = upload_user_image(img_bytes)
        except Exception:
            azure_url = None
        return {
            **state,
            "image_tags": [],
            "image_caption": "",
            "image_azure_url": azure_url
        }

    try:
        result = cv_client.analyze(
            image_data=img_bytes,
            visual_features=[
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.CAPTION
            ]
        )

        tags = [tag.name for tag in result.tags.list if tag.confidence > 0.7] if result.tags else []
        caption = result.caption.text if result.caption else ""
        
        try:
            azure_url = upload_user_image(img_bytes)
        except Exception:
            azure_url = None

        return {
            **state,
            "image_tags": tags,
            "image_caption": caption,
            # Upload to Azure Blob for logging/audit
            "image_azure_url": azure_url
        }
    except Exception as e:
        print(f"Error during image analysis: {e}")
        try:
            azure_url = upload_user_image(img_bytes)
        except Exception:
            azure_url = None
            
        return {
            **state,
            "image_tags": [],
            "image_caption": "",
            "image_azure_url": azure_url
        }
