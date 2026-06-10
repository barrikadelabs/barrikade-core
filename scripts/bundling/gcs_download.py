"""
Download models from Google Cloud Storage to core/models/.

This script downloads models from a PUBLIC GCS bucket to the local core/models/ directory,
with support for downloading specific layers or archived versions.

Usage:
    python scripts/bundling/gcs_download.py [--bucket my-bucket] [--layers layer_b,layer_c] [--version latest]
    
    --bucket:       GCS bucket name
    --layers:       Comma-separated layer names to download (default: all)
    --version:      'latest' for current models or 'archives' for archived versions
    --archive-old:  Backup current models before downloading (default: true)
    --validate:     Validate downloaded models (default: true)
    
Note:
    This script uses anonymous access for public buckets.
    No credentials are required.
"""

import argparse
import os
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Import GCS utilities
sys.path.insert(0, str(Path(__file__).parent))
import gcs_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORE_DIR = REPO_ROOT / "core"
MODELS_DIR = CORE_DIR / "models"
GCS_MODELS_PREFIX = "models"


def backup_local_models(layer_name: str) -> bool:
    """Backup current local models to a timestamped archive."""
    layer_dir = MODELS_DIR / layer_name
    
    if not layer_dir.exists():
        return True
    
    # Check if there are files to backup
    files = [f for f in layer_dir.rglob("*") if f.is_file() and "/archives/" not in str(f)]
    
    if not files:
        logger.info(f"  No current models to backup for {layer_name}")
        return True
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archives_dir = layer_dir / "archives"
        archives_dir.mkdir(exist_ok=True)
        
        backup_dir = archives_dir / f"backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"  Backing up to {backup_dir.relative_to(REPO_ROOT)}")
        
        for file_path in files:
            rel_path = file_path.relative_to(layer_dir)
            dest_path = backup_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_path)
        
        logger.info(f"  ✓ Backed up {len(files)} file(s)")
        return True
    
    except Exception as e:
        logger.error(f"  ✗ Backup failed: {e}")
        return False


def list_gcs_models(bucket_name: str, layer_name: str, version: str = "latest") -> List[dict]:
    """List available models in GCS for a layer."""
    try:
        client = gcs_utils.get_gcs_client(anonymous_only=True)
        
        if version == "latest":
            prefix = f"{GCS_MODELS_PREFIX}/{layer_name}/"
            exclude_archives = True
        elif version == "archives":
            prefix = f"{GCS_MODELS_PREFIX}/{layer_name}/archives/"
            exclude_archives = False
        else:
            raise ValueError(f"Unknown version: {version}")
        
        blobs = []
        for blob in client.list_blobs(bucket_name, prefix=prefix):
            if exclude_archives and "/archives/" in blob.name:
                continue
            if blob.name.endswith("/"):
                continue
            blobs.append({
                "name": blob.name,
                "size": blob.size,
            })
        
        return blobs
    
    except Exception as e:
        logger.error(f"Error listing GCS models: {e}")
        return []


def download_layer(
    bucket_name: str,
    layer_name: str,
    version: str = "latest",
    archive_old: bool = True,
    validate: bool = True,
    max_workers: int = 8,
):
    """
    Download all all files for a given layer from GCS in parallel.
    
    Returns:
        Dictionary with download results
    """
    result = {
        "layer": layer_name,
        "success": False,
        "files_downloaded": 0,
        "errors": [],
    }
    
    layer_dir = MODELS_DIR / layer_name
    layer_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nLayer {layer_name.upper()} (version: {version})")
    
    # Backup existing models
    if archive_old:
        backup_local_models(layer_name)
    
    # List GCS models
    gcs_files = list_gcs_models(bucket_name, layer_name, version)
    
    if not gcs_files:
        result["errors"].append(f"No models found in GCS for {layer_name} ({version})")
        return result
    
    logger.info(f"  Found {len(gcs_files)} file(s) in GCS")
    
    # Download files in parallel
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        lock = threading.Lock()
        
        def download_single(gcs_file_info):
            gcs_path = gcs_file_info["name"]
            gcs_size = gcs_file_info["size"]
            parts = gcs_path.split("/")
            rel_path = "/".join(parts[2:])  # Skip "models" and layer_name
            local_path = layer_dir / rel_path
            
            # Fast-path: Check if existing file is up-to-date and reuse it
            if local_path.exists() and local_path.stat().st_size == gcs_size:
                logger.info(f"  Skipping (up-to-date): {gcs_path}")
                with lock:
                    result["files_downloaded"] += 1
                return
            
            try:
                logger.info(f"  Downloading: {gcs_path}")
                gcs_utils.download_file_from_gcs(bucket_name, gcs_path, local_path)
                with lock:
                    result["files_downloaded"] += 1
            except Exception as e:
                error_msg = f"Failed to download {gcs_path}: {e}"
                logger.error(f"  ✗ {error_msg}")
                with lock:
                    result["errors"].append(error_msg)
        
        logger.info(f"  Downloading files using {max_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(download_single, gcs_file) for gcs_file in gcs_files]
            for future in as_completed(futures):
                future.result()
        
        result["success"] = len(result["errors"]) == 0
        
    except Exception as e:
        result["errors"].append(str(e))
    
    # Validate downloaded files
    if validate and result["success"]:
        files = [f for f in layer_dir.rglob("*") if f.is_file() and "/archives/" not in str(f)]
        if files:
            logger.info(f"  ✓ Downloaded and verified {len(files)} file(s)")
        else:
            result["success"] = False
            result["errors"].append("No files found after download")
    
    return result


def download_all_layers(
    bucket_name: str,
    layers: Optional[List[str]] = None,
    version: str = "latest",
    archive_old: bool = True,
    validate: bool = True,
    max_workers: int = 8,
) -> Dict[str, Dict]:
    """Download all or specified layers in parallel."""
    
    if layers is None:
        layers = ["layer_b", "layer_c", "layer_d", "layer_e"]
    
    results = {}
    
    for layer in layers:
        if layer not in ["layer_b", "layer_c", "layer_d", "layer_e"]:
            logger.warning(f"Unknown layer: {layer}")
            continue
        
        results[layer] = download_layer(
            bucket_name,
            layer,
            version=version,
            archive_old=archive_old,
            validate=validate,
            max_workers=max_workers,
        )
    
    return results


def validate_bucket_access(bucket_name: str) -> bool:
    """Validate that we can access the GCS bucket."""
    try:
        logger.info(f"Validating access to gs://{bucket_name}")
        client = gcs_utils.get_gcs_client(anonymous_only=True)
        bucket = client.bucket(bucket_name)
        logger.info(f"✓ Successfully authenticated to bucket")
        return True
    except Exception as e:
        logger.error(f"✗ Cannot access bucket: {e}")
        return False


def generate_download_manifest(
    bucket_name: str,
    results: Dict[str, Dict],
) -> Dict:
    """Generate a manifest of downloaded models."""
    
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "bucket": bucket_name,
        "models_prefix": GCS_MODELS_PREFIX,
        "layers": {},
    }
    
    for layer, result in results.items():
        manifest["layers"][layer] = {
            "success": result["success"],
            "files_downloaded": result["files_downloaded"],
            "errors": result["errors"],
        }
    
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Download models from Google Cloud Storage to core/models/"
    )
    parser.add_argument(
        "--bucket",
        default = "barrikade-bundles",
        help="GCS bucket name",
    )
    parser.add_argument(
        "--layers",
        help="Comma-separated layer names to download (default: all)",
    )
    parser.add_argument(
        "--version",
        choices=["latest", "archives"],
        default="latest",
        help="Version to download: 'latest' for current models, 'archives' for older versions (default: latest)",
    )
    parser.add_argument(
        "--archive-old",
        action="store_true",
        default=True,
        help="Backup current models before downloading (default: true)",
    )
    parser.add_argument(
        "--no-archive-old",
        dest="archive_old",
        action="store_false",
        help="Do not backup current models",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate downloaded models (default: true)",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Do not validate downloaded models",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Save download manifest to this file (JSON)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("BARRIKADA_MAX_DOWNLOAD_WORKERS", "8")),
        help="Number of concurrent download workers (default: 8)",
    )
    
    args = parser.parse_args()
    
    # Validate bucket access
    if not validate_bucket_access(args.bucket):
        return 1
    
    # Parse layers argument
    layers = None
    if args.layers:
        layers = [l.strip() for l in args.layers.split(",")]
    
    logger.info(f"Models directory: {MODELS_DIR.relative_to(REPO_ROOT)}")
    
    # Download layers
    results = download_all_layers(
        args.bucket,
        layers=layers,
        version=args.version,
        archive_old=args.archive_old,
        validate=args.validate,
        max_workers=args.max_workers,
    )
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*60}")
    
    total_files = 0
    total_success = 0
    
    for layer, result in results.items():
        status = "✓" if result["success"] else "✗"
        logger.info(f"{status} {layer}: {result['files_downloaded']} file(s) downloaded")
        
        if result["errors"]:
            for error in result["errors"]:
                logger.error(f"  → {error}")
        
        total_files += result["files_downloaded"]
        if result["success"]:
            total_success += 1
    
    logger.info(f"\nTotal: {total_success}/{len(results)} layers successful")
    
    # Save manifest
    if args.manifest:
        manifest = generate_download_manifest(args.bucket, results)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest saved to: {args.manifest}")
    
    return 0 if total_success == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
