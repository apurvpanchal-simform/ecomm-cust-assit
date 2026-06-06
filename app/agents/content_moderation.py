import os
import base64
import requests
from typing import Any

def content_moderation_node(state: Any) -> dict:
    """Reject NSFW or policy-violating images before any processing."""
    if not state.get("image_base64"):
        return {"image_is_safe": True}

    MODERATOR_ENDPOINT = os.environ.get("AZURE_CONTENT_MODERATOR_ENDPOINT")
    MODERATOR_KEY = os.environ.get("AZURE_CONTENT_MODERATOR_KEY")

    if not MODERATOR_ENDPOINT or not MODERATOR_KEY or MODERATOR_KEY.startswith("<paste-your"):
        # Fallback if keys are not provided
        return {"image_is_safe": True}

    try:
        img_base64 = state["image_base64"]
        # ensure endpoint doesn't have trailing slash
        endpoint = MODERATOR_ENDPOINT.rstrip('/')
        
        # We will attempt the modern Azure AI Content Safety API first
        url = f"{endpoint}/contentsafety/image:analyze?api-version=2023-10-01"
        
        response = requests.post(
            url,
            headers={
                "Ocp-Apim-Subscription-Key": MODERATOR_KEY,
                "Content-Type": "application/json"
            },
            json={
                "image": {"content": img_base64}
            }
        )
        response.raise_for_status()
        result = response.json()
        
        # Content Safety API returns 'categoriesAnalysis' with severity 0, 2, 4, 6
        is_safe = True
        for category in result.get("categoriesAnalysis", []):
            if category.get("severity", 0) > 0:
                is_safe = False
                break

        if not is_safe:
            return {
                "image_is_safe": False,
                "messages": [{"role": "assistant", "content": "I'm unable to process this image as it violates our content policy. Please upload a product photo."}]
            }

        return {"image_is_safe": True}
    except Exception as e:
        print(f"Content moderation error: {e}")
        # If API fails, default to safe so we don't break developer flows entirely.
        return {"image_is_safe": True}
