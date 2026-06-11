import os
from azure.storage.blob import BlobServiceClient


def get_blob_client_service():
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string or connection_string.startswith(
        "DefaultEndpointsProtocol=https;AccountName=..."
    ):
        pass

    # In case it's not set, we'll try to create it anyway so it fails natively
    return BlobServiceClient.from_connection_string(connection_string or "")


def get_container_name():
    return os.environ.get("AZURE_STORAGE_CONTAINER", "product-images")


def _ensure_container_exists(client: BlobServiceClient, container: str):
    container_client = client.get_container_client(container)
    if not container_client.exists():
        try:
            # First try to create it with public access so images are viewable
            container_client.create_container(public_access="blob")
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Could not create public container, falling back to private: {e}"
            )
            try:
                # Fallback to private container
                container_client.create_container()
            except Exception as e2:
                logging.getLogger(__name__).warning(
                    f"Failed to create private container: {e2}"
                )


def upload_product_image(image_bytes: bytes, product_id: str, ext: str = "jpg") -> str:
    """Upload product image, return public CDN URL."""
    client = get_blob_client_service()
    container = get_container_name()
    _ensure_container_exists(client, container)
    blob_name = f"catalog/{product_id}.{ext}"
    blob_client = client.get_blob_client(container=container, blob=blob_name)
    blob_client.upload_blob(image_bytes, overwrite=True)
    return blob_client.url
