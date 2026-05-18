"""
GCS utilities for Barrikada model hosting.

Provides helpers for Google Cloud Storage authentication, listing, and metadata operations.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def get_gcs_client(anonymous_only: bool = False):
    """
    Initialize and return a Google Cloud Storage client.
    
    Args:
        anonymous_only: If True, use only anonymous access (for public buckets).
                       If False, attempt authenticated access first (for uploads/private buckets).
    
    Returns:
        google.cloud.storage.Client: GCS client (authenticated or anonymous)
        
    Raises:
        ImportError: If google-cloud-storage is not installed
        ValueError: If authentication fails
    """
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError(
            "google-cloud-storage is required. Install with: pip install google-cloud-storage"
        )
    
    # For public bucket reads, use anonymous access only
    if anonymous_only:
        try:
            from google.auth.credentials import AnonymousCredentials
            from google.cloud import storage
            credentials = AnonymousCredentials()
            client = storage.Client(credentials=credentials, project="anonymouse-project")
            logger.info("Using anonymous access (public bucket read)")
            return client
        except Exception as e:
            raise ValueError(f"Failed to create anonymous GCS client: {e}")
    
    # For authenticated operations (uploads, private buckets), use credentials
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    try:
        if creds_path:
            if not Path(creds_path).exists():
                raise ValueError(f"Credentials file not found: {creds_path}")
            logger.info(f"Using credentials from: {creds_path}")
        else:
            logger.debug("Attempting to use Application Default Credentials (ADC)")
        
        client = storage.Client()
        # Verify client can authenticate
        _ = client.project
        logger.info(f"Successfully authenticated to GCS project: {client.project}")
        return client
    except Exception as e:
        raise ValueError(f"Failed to authenticate to GCS: {e}")


def get_bucket(bucket_name: str):
    """
    Get a GCS bucket object.
    
    Args:
        bucket_name: Name of the GCS bucket
        
    Returns:
        google.cloud.storage.Bucket: Bucket object
        
    Raises:
        ValueError: If bucket cannot be accessed
    """
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("google-cloud-storage is required")
    
    client = get_gcs_client()
    
    try:
        bucket = client.bucket(bucket_name)
        return bucket
    except Exception as e:
        raise ValueError(f"Cannot access bucket '{bucket_name}': {e}")


def list_models_in_bucket(bucket_name: str, prefix: str = "models/") -> Dict[str, List[str]]:
    """
    List all models in a GCS bucket grouped by layer.
    
    Args:
        bucket_name: Name of the GCS bucket
        prefix: Prefix to search under (default: "models/")
        
    Returns:
        Dictionary mapping layer names to lists of object paths
    """
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("google-cloud-storage is required")
    
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    
    models = {
        "layer_b": [],
        "layer_c": [],
        "layer_d": [],
        "layer_e": [],
        "archives": [],
    }
    
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    
    for blob in blobs:
        path = blob.name
        if "/archives/" in path:
            models["archives"].append(path)
        elif "layer_b" in path:
            models["layer_b"].append(path)
        elif "layer_c" in path:
            models["layer_c"].append(path)
        elif "layer_d" in path:
            models["layer_d"].append(path)
        elif "layer_e" in path:
            models["layer_e"].append(path)
    
    return models


def get_blob_metadata(bucket_name: str, blob_path: str) -> Dict[str, Any]:
    """
    Get metadata for a GCS blob.
    
    Args:
        bucket_name: Name of the GCS bucket
        blob_path: Path to the blob in the bucket
        
    Returns:
        Dictionary with blob metadata (size, time_created, updated, etc.)
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    if not blob.exists():
        raise ValueError(f"Blob not found: {blob_path}")
    
    return {
        "name": blob.name,
        "size": blob.size,
        "created": blob.time_created.isoformat() if blob.time_created else None,
        "updated": blob.updated.isoformat() if blob.updated else None,
        "content_type": blob.content_type,
        "md5": blob.md5_hash,
    }


def upload_file_to_gcs(
    local_path: Path,
    bucket_name: str,
    blob_path: str,
) -> bool:
    """
    Upload a file to GCS.
    
    Args:
        local_path: Local file path to upload
        bucket_name: Name of the GCS bucket
        blob_path: Destination path in bucket
        
    Returns:
        True if successful
        
    Raises:
        ValueError: If file doesn't exist or upload fails
    """
    if not local_path.exists():
        raise ValueError(f"File not found: {local_path}")
    
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    file_size = local_path.stat().st_size
    logger.info(f"Uploading {local_path.name} ({file_size:,} bytes) to gs://{bucket_name}/{blob_path}")
    
    try:
        blob.upload_from_filename(str(local_path))
        logger.info(f"Successfully uploaded: gs://{bucket_name}/{blob_path}")
        return True
    except Exception as e:
        raise ValueError(f"Upload failed: {e}")


def download_file_from_gcs(
    bucket_name: str,
    blob_path: str,
    local_path: Path,
) -> bool:
    """
    Download a file from public GCS bucket using direct HTTP access.
    
    Args:
        bucket_name: Name of the GCS bucket (must be publicly readable)
        blob_path: Path to blob in bucket
        local_path: Local destination path
        
    Returns:
        True if successful
        
    Raises:
        ValueError: If download fails
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading gs://{bucket_name}/{blob_path} to {local_path}")
    
    try:
        import requests
        # Use direct HTTPS download for public buckets (no authentication needed)
        url = f"https://storage.googleapis.com/{bucket_name}/{blob_path}"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 404:
            raise ValueError(f"Blob not found in GCS: {blob_path}")
        elif response.status_code == 403:
            raise ValueError(f"Access denied to bucket (not publicly readable): {bucket_name}")
        elif response.status_code != 200:
            raise ValueError(f"HTTP error {response.status_code}: {response.reason}")
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Successfully downloaded: {local_path}")
        return True
    except Exception as e:
        raise ValueError(f"Download failed: {e}")


def blob_exists(bucket_name: str, blob_path: str) -> bool:
    """
    Check if a blob exists in GCS.
    
    Args:
        bucket_name: Name of the GCS bucket
        blob_path: Path to blob in bucket
        
    Returns:
        True if blob exists, False otherwise
    """
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.exists()
    except Exception as e:
        logger.error(f"Error checking blob existence: {e}")
        return False


def copy_blob_in_gcs(
    bucket_name: str,
    source_blob_path: str,
    dest_blob_path: str,
) -> bool:
    """
    Copy a blob within the same GCS bucket.
    
    Args:
        bucket_name: Name of the GCS bucket
        source_blob_path: Source blob path
        dest_blob_path: Destination blob path
        
    Returns:
        True if successful
        
    Raises:
        ValueError: If copy fails
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    source_blob = bucket.blob(source_blob_path)
    
    if not source_blob.exists():
        raise ValueError(f"Source blob not found: {source_blob_path}")
    
    try:
        logger.info(f"Copying gs://{bucket_name}/{source_blob_path} to {dest_blob_path}")
        bucket.copy_blob(source_blob, bucket, dest_blob_path)
        logger.info(f"Successfully copied")
        return True
    except Exception as e:
        raise ValueError(f"Copy failed: {e}")


def delete_blob(bucket_name: str, blob_path: str) -> bool:
    """
    Delete a blob from GCS.
    
    Args:
        bucket_name: Name of the GCS bucket
        blob_path: Path to blob to delete
        
    Returns:
        True if successful or blob didn't exist
    """
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        if blob.exists():
            blob.delete()
            logger.info(f"Deleted: gs://{bucket_name}/{blob_path}")
        return True
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return False
