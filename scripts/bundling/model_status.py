"""
Show status of local models vs. GCS models.

Displays information about:
- Local models: what's in core/models/ and archives/
- GCS models: what's in the bucket
- Last update times
- Comparison between local and GCS versions

Usage:
    python scripts/bundling/model_status.py --bucket my-bucket [--layers layer_b,layer_c] [--compare]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict
from datetime import datetime

# Import GCS utilities
sys.path.insert(0, str(Path(__file__).parent))
import gcs_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "core" / "models"
GCS_MODELS_PREFIX = "models"


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0 # type: ignore
    return f"{size_bytes:.1f} TB"


def get_local_status(layer_name: str) -> Dict:
    """Get status of local models for a layer."""
    layer_dir = MODELS_DIR / layer_name
    
    if not layer_dir.exists():
        return {
            "exists": False,
            "current_files": [],
            "current_size": 0,
            "archives": {},
        }
    
    # Current models (not in archives)
    current_files = []
    current_size = 0
    
    for file_path in layer_dir.rglob("*"):
        if file_path.is_file() and "/archives/" not in str(file_path):
            current_files.append(file_path.name)
            current_size += file_path.stat().st_size
    
    # Archived models
    archives = {}
    archives_dir = layer_dir / "archives"
    
    if archives_dir.exists():
        for archive_subdir in archives_dir.iterdir():
            if archive_subdir.is_dir():
                archive_files = []
                archive_size = 0
                
                for file_path in archive_subdir.rglob("*"):
                    if file_path.is_file():
                        archive_files.append(file_path.name)
                        archive_size += file_path.stat().st_size
                
                if archive_files:
                    mtime = datetime.fromtimestamp(archive_subdir.stat().st_mtime)
                    archives[archive_subdir.name] = {
                        "files": archive_files,
                        "size": archive_size,
                        "mtime": mtime.isoformat(),
                    }
    
    return {
        "exists": True,
        "current_files": current_files,
        "current_size": current_size,
        "archives": archives,
    }


def get_gcs_status(bucket_name: str, layer_name: str) -> Dict:
    """Get status of models in GCS for a layer."""
    try:
        client = gcs_utils.get_gcs_client()
        
        # Current models
        prefix = f"{GCS_MODELS_PREFIX}/{layer_name}/"
        current_files = []
        current_size = 0
        latest_update = None
        
        for blob in client.list_blobs(bucket_name, prefix=prefix):
            if "/archives/" in blob.name or blob.name.endswith("/"):
                continue
            
            current_files.append(blob.name.split("/")[-1])
            current_size += blob.size
            
            if blob.updated:
                if latest_update is None or blob.updated > latest_update:
                    latest_update = blob.updated
        
        # Archives
        archives = {}
        archive_prefix = f"{GCS_MODELS_PREFIX}/{layer_name}/archives/"
        
        current_archive = None
        archive_files = []
        archive_size = 0
        
        for blob in client.list_blobs(bucket_name, prefix=archive_prefix):
            if blob.name.endswith("/"):
                continue
            
            # Parse archive folder name
            parts = blob.name.split("/")
            if len(parts) >= 5:
                archive_name = parts[3]
            else:
                continue
            
            if current_archive != archive_name:
                if current_archive and archive_files:
                    archives[current_archive] = {
                        "files": archive_files,
                        "size": archive_size,
                    }
                current_archive = archive_name
                archive_files = []
                archive_size = 0
            
            archive_files.append(blob.name.split("/")[-1])
            archive_size += blob.size
        
        if current_archive and archive_files:
            archives[current_archive] = {
                "files": archive_files,
                "size": archive_size,
            }
        
        return {
            "exists": len(current_files) > 0 or len(archives) > 0,
            "current_files": current_files,
            "current_size": current_size,
            "latest_update": latest_update.isoformat() if latest_update else None,
            "archives": archives,
        }
    
    except Exception as e:
        logger.warning(f"Error accessing GCS: {e}")
        return {
            "exists": False,
            "current_files": [],
            "current_size": 0,
            "error": str(e),
        }


def print_layer_status(layer_name: str, local_status: Dict, gcs_status: Dict):
    """Print status for a single layer."""
    print(f"\n{'='*70}")
    print(f"Layer {layer_name.upper()}")
    print(f"{'='*70}")
    
    # Local status
    print(f"\nLocal (core/models/{layer_name}/):")
    if local_status["exists"]:
        print(f"  Current: {len(local_status['current_files'])} file(s), {format_size(local_status['current_size'])}")
        if local_status["current_files"]:
            for fname in sorted(local_status["current_files"])[:5]:
                print(f"    - {fname}")
            if len(local_status["current_files"]) > 5:
                print(f"    ... and {len(local_status['current_files']) - 5} more")
        
        if local_status["archives"]:
            print(f"  Archives: {len(local_status['archives'])} version(s)")
            for archive_name, archive_info in sorted(local_status["archives"].items(), reverse=True)[:3]:
                print(f"    - {archive_name}: {len(archive_info['files'])} file(s), {format_size(archive_info['size'])}")
                print(f"      Modified: {archive_info['mtime']}")
    else:
        print(f"  ✗ No models found locally")
    
    # GCS status
    print(f"\nGCS (gs://bucket/models/{layer_name}/):")
    if gcs_status.get("error"):
        print(f"  ✗ Error: {gcs_status['error']}")
    elif gcs_status["exists"]:
        print(f"  Current: {len(gcs_status['current_files'])} file(s), {format_size(gcs_status['current_size'])}")
        if gcs_status["current_files"]:
            for fname in sorted(gcs_status["current_files"])[:5]:
                print(f"    - {fname}")
            if len(gcs_status["current_files"]) > 5:
                print(f"    ... and {len(gcs_status['current_files']) - 5} more")
        
        if gcs_status.get("latest_update"):
            print(f"  Latest update: {gcs_status['latest_update']}")
        
        if gcs_status.get("archives"):
            print(f"  Archives: {len(gcs_status['archives'])} version(s)")
            for archive_name, archive_info in sorted(gcs_status["archives"].items(), reverse=True)[:3]:
                print(f"    - {archive_name}: {len(archive_info['files'])} file(s)")
    else:
        print(f"  ✗ No models found on GCS")
    
    # Comparison
    if local_status["exists"] and gcs_status["exists"]:
        local_count = len(local_status["current_files"])
        gcs_count = len(gcs_status["current_files"])
        
        if local_count == gcs_count:
            print(f"\n  Status: ✓ Local and GCS in sync")
        else:
            print(f"\n  Status: ⚠ Mismatch - Local: {local_count} files, GCS: {gcs_count} files")


def main():
    parser = argparse.ArgumentParser(
        description="Show status of local and GCS models"
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="GCS bucket name",
    )
    parser.add_argument(
        "--layers",
        help="Comma-separated layer names (default: all)",
    )
    
    args = parser.parse_args()
    
    # Validate bucket access
    logger.info(f"Connecting to gs://{args.bucket}...\n")
    
    try:
        gcs_utils.get_gcs_client()
    except Exception as e:
        logger.error(f"✗ Cannot access GCS: {e}")
        return 1
    
    # Parse layers
    if args.layers:
        layers = [l.strip() for l in args.layers.split(",")]
    else:
        layers = ["layer_b", "layer_c", "layer_d", "layer_e"]
    
    # Get and print status for each layer
    for layer_name in layers:
        local_status = get_local_status(layer_name)
        gcs_status = get_gcs_status(args.bucket, layer_name)
        print_layer_status(layer_name, local_status, gcs_status)
    
    print(f"\n{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
