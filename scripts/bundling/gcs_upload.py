"""
Upload bundled models from core/models/ to Google Cloud Storage.

This script uploads all models in core/models/ to a GCS bucket, with support for
archiving the previous version.

Usage:
    python scripts/bundling/gcs_upload.py --bucket my-bucket [--project my-project] [--layers layer_b,layer_c] [--dry-run]
    
    --bucket:       GCS bucket name
    --project:      GCP project ID (optional, uses gcloud default if not specified)
    --layers:       Comma-separated layer names to upload (default: all)
    --archive:      Archive previous version before uploading (default: true)
    --dry-run:      Show what would be uploaded without making changes
    
Environment:
    GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON (or use Application Default Credentials)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Import GCS utilities
import sys
sys.path.insert(0, str(Path(__file__).parent))
import gcs_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPO_ROOT = Path(__file__).parent.parent
CORE_DIR = REPO_ROOT / "core"
MODELS_DIR = CORE_DIR / "models"

# GCS prefix where models are stored
GCS_MODELS_PREFIX = "models"


def get_local_files(layer_name: str, exclude_archives: bool = True) -> List[Path]:
    """Get all files for a given layer."""
    layer_dir = MODELS_DIR / layer_name
    
    if not layer_dir.exists():
        return []
    
    files = []
    for file_path in layer_dir.rglob("*"):
        if file_path.is_file():
            if exclude_archives and "/archives/" in str(file_path):
                continue
            files.append(file_path)
    
    return files


def archive_previous_version(
    bucket_name: str,
    layer_name: str,
    dry_run: bool = False,
) -> bool:
    """
    Move the current version of a layer in GCS to archives.
    
    Returns:
        True if archive was created or no files to archive
    """
    try:
        client = gcs_utils.get_gcs_client()
        
        # List current models for this layer
        prefix = f"{GCS_MODELS_PREFIX}/{layer_name}/"
        blobs = list(client.list_blobs(bucket_name, prefix=prefix, delimiter="/"))
        
        if not blobs:
            logger.info(f"  No previous version to archive for {layer_name}")
            return True
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_prefix = f"{GCS_MODELS_PREFIX}/{layer_name}/archives/backup_{timestamp}/"
        
        logger.info(f"  Archiving previous version to {archive_prefix}")
        
        bucket = client.bucket(bucket_name)
        
        for blob in blobs:
            if "/archives/" in blob.name:
                continue
            
            archive_path = blob.name.replace(
                f"{GCS_MODELS_PREFIX}/{layer_name}/",
                archive_prefix,
            )
            
            if not dry_run:
                try:
                    gcs_utils.copy_blob_in_gcs(bucket_name, blob.name, archive_path)
                    logger.debug(f"    Archived: {blob.name} → {archive_path}")
                except Exception as e:
                    logger.warning(f"    Failed to archive {blob.name}: {e}")
            else:
                logger.info(f"    [DRY RUN] Would archive: {blob.name} → {archive_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error archiving previous version: {e}")
        return False


def upload_layer(
    bucket_name: str,
    layer_name: str,
    dry_run: bool = False,
    archive: bool = True,
):
    """
    Upload all files for a given layer.
    
    Returns:
        Dictionary with upload results
    """
    result = {
        "layer": layer_name,
        "success": False,
        "archived": False,
        "files_uploaded": 0,
        "errors": [],
    }
    
    layer_dir = MODELS_DIR / layer_name
    
    if not layer_dir.exists():
        result["errors"].append(f"Layer directory not found: {layer_dir}")
        return result
    
    files = get_local_files(layer_name)
    
    if not files:
        result["errors"].append(f"No files found in {layer_dir}")
        return result
    
    logger.info(f"\nLayer {layer_name.upper()}: {len(files)} file(s)")
    
    # Archive previous version
    if archive:
        result["archived"] = archive_previous_version(bucket_name, layer_name, dry_run)
    
    # Upload new files
    try:
        for file_path in files:
            rel_path = file_path.relative_to(layer_dir)
            blob_path = f"{GCS_MODELS_PREFIX}/{layer_name}/{rel_path}"
            
            file_size = file_path.stat().st_size
            
            if dry_run:
                logger.info(f"  [DRY RUN] Would upload: {rel_path} ({file_size:,} bytes)")
                result["files_uploaded"] += 1
            else:
                try:
                    gcs_utils.upload_file_to_gcs(file_path, bucket_name, blob_path)
                    result["files_uploaded"] += 1
                except Exception as e:
                    error_msg = f"Failed to upload {rel_path}: {e}"
                    logger.error(f"  ✗ {error_msg}")
                    result["errors"].append(error_msg)
        
        result["success"] = len(result["errors"]) == 0
        
    except Exception as e:
        result["errors"].append(str(e))
    
    return result


def upload_all_layers(
    bucket_name: str,
    layers: Optional[List[str]] = None,
    dry_run: bool = False,
    archive: bool = True,
) -> Dict[str, Dict]:
    """Upload all or specified layers."""
    
    if layers is None:
        layers = ["layer_b", "layer_c", "layer_d", "layer_e"]
    
    results = {}
    
    for layer in layers:
        if layer not in ["layer_b", "layer_c", "layer_d", "layer_e"]:
            logger.warning(f"Unknown layer: {layer}")
            continue
        
        results[layer] = upload_layer(bucket_name, layer, dry_run, archive)
    
    return results


def validate_bucket_access(bucket_name: str) -> bool:
    """Validate that we can access the GCS bucket."""
    try:
        logger.info(f"Validating access to gs://{bucket_name}")
        bucket = gcs_utils.get_bucket(bucket_name)
        logger.info(f"✓ Successfully authenticated to bucket")
        return True
    except Exception as e:
        logger.error(f"✗ Cannot access bucket: {e}")
        return False


def generate_upload_manifest(
    bucket_name: str,
    results: Dict[str, Dict],
) -> Dict:
    """Generate a manifest of uploaded models."""
    
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "bucket": bucket_name,
        "models_prefix": GCS_MODELS_PREFIX,
        "layers": {},
    }
    
    for layer, result in results.items():
        manifest["layers"][layer] = {
            "success": result["success"],
            "files_uploaded": result["files_uploaded"],
            "archived": result["archived"],
            "errors": result["errors"],
        }
    
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Upload bundled models to Google Cloud Storage"
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="GCS bucket name",
    )
    parser.add_argument(
        "--project",
        help="GCP project ID",
    )
    parser.add_argument(
        "--layers",
        help="Comma-separated layer names to upload (default: all)",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        default=True,
        help="Archive previous version before uploading (default: true)",
    )
    parser.add_argument(
        "--no-archive",
        dest="archive",
        action="store_false",
        help="Do not archive previous version",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without making changes",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Save upload manifest to this file (JSON)",
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("[DRY RUN] No files will be uploaded")
    
    # Validate bucket access
    if not validate_bucket_access(args.bucket):
        return 1
    
    # Parse layers argument
    layers = None
    if args.layers:
        layers = [l.strip() for l in args.layers.split(",")]
    
    logger.info(f"Models directory: {MODELS_DIR.relative_to(REPO_ROOT)}")
    
    # Upload layers
    results = upload_all_layers(
        args.bucket,
        layers=layers,
        dry_run=args.dry_run,
        archive=args.archive,
    )
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("UPLOAD SUMMARY")
    logger.info(f"{'='*60}")
    
    total_files = 0
    total_success = 0
    
    for layer, result in results.items():
        status = "✓" if result["success"] else "✗"
        logger.info(f"{status} {layer}: {result['files_uploaded']} file(s) uploaded")
        
        if result["errors"]:
            for error in result["errors"]:
                logger.error(f"  → {error}")
        
        total_files += result["files_uploaded"]
        if result["success"]:
            total_success += 1
    
    logger.info(f"\nTotal: {total_success}/{len(results)} layers successful")
    
    # Save manifest
    if args.manifest:
        manifest = generate_upload_manifest(args.bucket, results)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest saved to: {args.manifest}")
    
    return 0 if total_success == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
