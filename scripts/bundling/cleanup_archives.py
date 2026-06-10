"""
Clean up old archived model versions.

Removes archived model versions beyond a specified threshold, keeping only
the N most recent versions on both local filesystem and GCS.

Usage:
    python scripts/bundling/cleanup_archives.py --keep 3 [--layers layer_b,layer_c] [--local] [--gcs] [--bucket my-bucket] [--dry-run]
    
    --keep:         Number of recent versions to keep (default: 3)
    --layers:       Comma-separated layer names to clean (default: all)
    --local:        Clean local archives (default: true if --gcs not specified)
    --gcs:          Clean GCS archives (requires --bucket)
    --bucket:       GCS bucket name
    --dry-run:      Show what would be deleted without making changes
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Dict

# Import GCS utilities
sys.path.insert(0, str(Path(__file__).parent))
import gcs_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "core" / "models"
GCS_MODELS_PREFIX = "models"


def get_archive_versions(layer_dir: Path) -> List[tuple]:
    """
    Get archive versions sorted by modification time (newest first).
    
    Returns:
        List of (archive_path, mtime) tuples
    """
    archives_dir = layer_dir / "archives"
    
    if not archives_dir.exists():
        return []
    
    versions = []
    for archive_subdir in archives_dir.iterdir():
        if archive_subdir.is_dir():
            mtime = archive_subdir.stat().st_mtime
            versions.append((archive_subdir, mtime))
    
    # Sort by mtime descending (newest first)
    versions.sort(key=lambda x: x[1], reverse=True)
    return versions


def clean_local_archives(layer_name: str, keep: int = 3, dry_run: bool = False) -> Dict:
    """Clean local archives for a layer."""
    result = {
        "layer": layer_name,
        "deleted_versions": 0,
        "freed_space": 0,
        "errors": [],
    }
    
    layer_dir = MODELS_DIR / layer_name
    
    if not layer_dir.exists():
        return result
    
    versions = get_archive_versions(layer_dir)
    
    if len(versions) <= keep:
        logger.info(f"{layer_name}: {len(versions)} versions, keeping all (threshold: {keep})")
        return result
    
    logger.info(f"{layer_name}: {len(versions)} versions, keeping {keep} most recent")
    
    # Delete old versions
    for old_version_path, _ in versions[keep:]:
        try:
            # Calculate size before deletion
            total_size = 0
            for file_path in old_version_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            if dry_run:
                logger.info(f"  [DRY RUN] Would delete: {old_version_path.name} ({format_size(total_size)})")
            else:
                logger.info(f"  Deleting: {old_version_path.name} ({format_size(total_size)})")
                shutil.rmtree(old_version_path)
                result["deleted_versions"] += 1
                result["freed_space"] += total_size
        
        except Exception as e:
            error_msg = f"Failed to delete {old_version_path.name}: {e}"
            logger.error(f"  ✗ {error_msg}")
            result["errors"].append(error_msg)
    
    return result


def get_gcs_archive_versions(bucket_name: str, layer_name: str) -> List[tuple]:
    """
    Get GCS archive versions sorted by name/timestamp (assuming naming pattern backup_YYYYMMDD_HHMMSS).
    
    Returns:
        List of (archive_prefix, timestamp_str) tuples
    """
    try:
        client = gcs_utils.get_gcs_client()
        
        archive_prefix = f"{GCS_MODELS_PREFIX}/{layer_name}/archives/"
        versions = {}
        
        for blob in client.list_blobs(bucket_name, prefix=archive_prefix):
            if blob.name.endswith("/"):
                continue
            
            # Parse archive folder name from path
            parts = blob.name.split("/")
            if len(parts) >= 5 and "backup_" in parts[3]:
                archive_name = parts[3]
                if archive_name not in versions:
                    versions[archive_name] = archive_prefix + archive_name
        
        # Sort by name descending (should put newer dates first if naming is consistent)
        sorted_versions = sorted(versions.items(), key=lambda x: x[0], reverse=True)
        return [(v[1], v[0]) for v in sorted_versions]
    
    except Exception as e:
        logger.error(f"Error listing GCS archives: {e}")
        return []


def clean_gcs_archives(
    bucket_name: str,
    layer_name: str,
    keep: int = 3,
    dry_run: bool = False,
) -> Dict:
    """Clean GCS archives for a layer."""
    result = {
        "layer": layer_name,
        "deleted_versions": 0,
        "freed_space": 0,
        "errors": [],
    }
    
    versions = get_gcs_archive_versions(bucket_name, layer_name)
    
    if len(versions) <= keep:
        logger.info(f"{layer_name}: {len(versions)} versions, keeping all (threshold: {keep})")
        return result
    
    logger.info(f"{layer_name}: {len(versions)} versions, keeping {keep} most recent")
    
    try:
        client = gcs_utils.get_gcs_client()
        bucket = client.bucket(bucket_name)
        
        # Delete old versions
        for archive_prefix, archive_name in versions[keep:]:
            try:
                # List files in this archive
                archive_blobs = list(client.list_blobs(bucket_name, prefix=archive_prefix + "/"))
                total_size = sum(blob.size for blob in archive_blobs if not blob.name.endswith("/"))
                
                if dry_run:
                    logger.info(f"  [DRY RUN] Would delete: {archive_name} ({format_size(total_size)})")
                else:
                    logger.info(f"  Deleting: {archive_name} ({format_size(total_size)})")
                    
                    # Delete all files in archive
                    for blob in archive_blobs:
                        if not blob.name.endswith("/"):
                            blob.delete()
                    
                    result["deleted_versions"] += 1
                    result["freed_space"] += total_size
            
            except Exception as e:
                error_msg = f"Failed to delete {archive_name}: {e}"
                logger.error(f"  ✗ {error_msg}")
                result["errors"].append(error_msg)
    
    except Exception as e:
        logger.error(f"Error accessing GCS: {e}")
        result["errors"].append(str(e))
    
    return result


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0 # type: ignore
    return f"{size_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old archived model versions"
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="Number of recent versions to keep (default: 3)",
    )
    parser.add_argument(
        "--layers",
        help="Comma-separated layer names (default: all)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Clean local archives",
    )
    parser.add_argument(
        "--gcs",
        action="store_true",
        help="Clean GCS archives (requires --bucket)",
    )
    parser.add_argument(
        "--bucket",
        help="GCS bucket name (required if --gcs specified)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without making changes",
    )
    
    args = parser.parse_args()
    
    # Determine what to clean
    clean_local = args.local
    clean_gcs_flag = args.gcs
    
    if not clean_local and not clean_gcs_flag:
        clean_local = True
    
    if clean_gcs_flag and not args.bucket:
        logger.error("--bucket is required when using --gcs")
        return 1
    
    # Parse layers
    if args.layers:
        layers = [l.strip() for l in args.layers.split(",")]
    else:
        layers = ["layer_b", "layer_c", "layer_d", "layer_e"]
    
    if args.dry_run:
        logger.info("[DRY RUN] No files will be deleted\n")
    
    # Clean archives
    results = {}
    
    if clean_local:
        logger.info("CLEANING LOCAL ARCHIVES")
        logger.info(f"Keeping {args.keep} most recent version(s)\n")
        
        for layer in layers:
            results[f"local_{layer}"] = clean_local_archives(layer, args.keep, args.dry_run)
    
    if clean_gcs_flag:
        logger.info("\nCLEANING GCS ARCHIVES")
        logger.info(f"Bucket: gs://{args.bucket}")
        logger.info(f"Keeping {args.keep} most recent version(s)\n")
        
        for layer in layers:
            results[f"gcs_{layer}"] = clean_gcs_archives(
                args.bucket, layer, args.keep, args.dry_run
            )
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("CLEANUP SUMMARY")
    logger.info(f"{'='*60}")
    
    total_freed = 0
    total_deleted = 0
    
    for name, result in results.items():
        if result["errors"]:
            logger.error(f"✗ {name}: {result['deleted_versions']} version(s) deleted")
            for error in result["errors"]:
                logger.error(f"  → {error}")
        else:
            logger.info(f"✓ {name}: {result['deleted_versions']} version(s) deleted, {format_size(result['freed_space'])} freed")
        
        total_deleted += result["deleted_versions"]
        total_freed += result["freed_space"]
    
    logger.info(f"\nTotal: {total_deleted} version(s) deleted, {format_size(total_freed)} freed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
