"""
Bundle models from scattered layer_*/outputs/ directories into core/models/ structure.

This script consolidates trained models from each layer's outputs directory into a
centralized core/models/ directory tree, organizing them by layer with support for
versioning via archives.

Usage:
    python scripts/bundling/bundle_models.py [--dry-run] [--archive-old]
    
    --dry-run:       Show what would be bundled without making changes
    --archive-old:   Move current models to archives before bundling new ones
"""

import argparse
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORE_DIR = REPO_ROOT / "core"
MODELS_DIR = CORE_DIR / "models"

@dataclass(frozen=True)
class LayerConfig:
    outputs_dir: Path
    target_dir: Path
    required_patterns: Tuple[str, ...]
    description: str


LAYER_CONFIGS: Dict[str, LayerConfig] = {
    "layer_b": LayerConfig(
        outputs_dir=CORE_DIR / "layer_b" / "signatures",
        target_dir=MODELS_DIR / "layer_b",
        required_patterns=(
            "embeddings/centroids.npy",
            "embeddings/benign_centroids.npy",
            "embeddings/cluster_radii.json",
            "embeddings/metadata.json",
            "embeddings/faiss_index.bin",
            "embeddings/benign_faiss_index.bin",
            "embeddings/prompt_encoder/",
            "embeddings/prompt_encoder_onnx/"
        ),
        description="Signature Engine (FAISS indices, embeddings)",
    ),
    "layer_c": LayerConfig(
        outputs_dir=CORE_DIR / "layer_c" / "outputs",
        target_dir=MODELS_DIR / "layer_c",
        required_patterns=(
            "classifier.joblib",
            "classifier.onnx",
            "calibrator.joblib",
            "encoder_onnx/",
        ),
        description="ML Classifier (XGBoost/sklearn models)",
    ),
    "layer_d": LayerConfig(
        outputs_dir=CORE_DIR / "layer_d" / "outputs",
        target_dir=MODELS_DIR / "layer_d",
        required_patterns=("model/", "onnx/", "tokenizer.json", "*.safetensors"),
        description="ModernBERT (Hugging Face model)",
    ),
    "layer_e": LayerConfig(
        outputs_dir=CORE_DIR / "layer_e" / "outputs",
        target_dir=MODELS_DIR / "layer_e",
        required_patterns=("qwen3guard-barrikade/", "*.json"),
        description="LLM Judge (bundled Hugging Face checkpoint)",
    ),
}


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    deduped: dict[Path, None] = {}
    for path in paths:
        deduped[path] = None
    return list(deduped.keys())


def get_model_files(source_dir: Path, patterns: Iterable[str]) -> List[Path]:
    """Find all files matching the given patterns."""
    files = []
    if not source_dir.exists():
        return files
    
    for pattern in patterns:
        if pattern.endswith("/"):
            # Directory pattern
            dir_name = pattern.rstrip("/")
            dir_path = source_dir / dir_name
            if dir_path.is_dir():
                files.extend(dir_path.rglob("*"))
        else:
            # File pattern
            files.extend(source_dir.glob(pattern))
            files.extend(source_dir.rglob(pattern))
    
    return sorted(_unique_paths(files))


def archive_existing_models(target_dir: Path) -> bool:
    """Move existing models in target_dir to archives subfolder."""
    if not target_dir.exists() or not list(target_dir.glob("*")):
        return False
    
    archives_dir = target_dir / "archives"
    archives_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_subdir = archives_dir / f"backup_{timestamp}"
    
    logger.info("Archiving existing models to %s", archive_subdir.relative_to(REPO_ROOT))
    
    # Move all non-archive items to archive
    for item in target_dir.iterdir():
        if item.is_dir() and item.name == "archives":
            continue
        
        archive_subdir.mkdir(parents=True, exist_ok=True)
        dest = archive_subdir / item.name
        
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
            shutil.rmtree(item)
        else:
            shutil.copy2(item, dest)
            item.unlink()
    
    return True


def bundle_layer(
    layer_name: str,
    config: LayerConfig,
    dry_run: bool = False,
    archive_old: bool = False,
) -> Tuple[bool, str]:
    """
    Bundle models for a single layer.
    
    Returns:
        (success, message)
    """
    outputs_dir = config.outputs_dir
    target_dir = config.target_dir
    patterns = config.required_patterns
    description = config.description
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find model files
    model_files = get_model_files(outputs_dir, patterns)
    
    if not model_files:
        return False, f"No model files found in {outputs_dir.relative_to(REPO_ROOT)}"
    
    logger.info("\n%s", "=" * 60)
    logger.info("Layer %s: %s", layer_name.upper(), description)
    logger.info("%s", "=" * 60)
    logger.info(
        "Found %s file(s) in %s",
        len(model_files),
        outputs_dir.relative_to(REPO_ROOT),
    )
    
    if archive_old:
        if not dry_run:
            archive_existing_models(target_dir)
        else:
            logger.info(
                "[DRY RUN] Would archive existing models in %s",
                target_dir.relative_to(REPO_ROOT),
            )
    
    # Copy/link model files
    for src_file in model_files:
        if src_file.is_file():
            # Preserve directory structure relative to outputs_dir
            rel_path = src_file.relative_to(outputs_dir)
            dest_file = target_dir / rel_path
            
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            if dry_run:
                logger.info(
                    "[DRY RUN] Would copy: %s",
                    src_file.relative_to(REPO_ROOT),
                )
                logger.info(
                    "          to: %s",
                    dest_file.relative_to(REPO_ROOT),
                )
            else:
                if not dest_file.exists() or dest_file.stat().st_mtime < src_file.stat().st_mtime:
                    shutil.copy2(src_file, dest_file)
                    logger.info("Copied: %s", rel_path)
                else:
                    logger.info("Skipped (up-to-date): %s", rel_path)
        elif src_file.is_dir() and src_file != target_dir / "archives":
            # Copy entire directory
            rel_path = src_file.relative_to(outputs_dir)
            dest_dir = target_dir / rel_path
            
            if dry_run:
                logger.info(
                    "[DRY RUN] Would copy directory: %s",
                    src_file.relative_to(REPO_ROOT),
                )
                logger.info(
                    "          to: %s",
                    dest_dir.relative_to(REPO_ROOT),
                )
            else:
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(src_file, dest_dir)
                logger.info("Copied directory: %s", rel_path)
    
    return True, f"Bundled {len(model_files)} file(s) for {layer_name}"


def validate_bundle() -> bool:
    """Validate that bundled models are present and accessible."""
    logger.info("\n%s", "=" * 60)
    logger.info("VALIDATION")
    logger.info("%s", "=" * 60)
    
    all_valid = True
    for layer_name, config in LAYER_CONFIGS.items():
        target_dir = config.target_dir
        
        if not target_dir.exists():
            logger.warning("%s: Target directory does not exist", layer_name)
            all_valid = False
            continue
        
        files = list(target_dir.glob("**/*"))
        files = [f for f in files if f.is_file() and f.parent.name != "archives"]
        
        if files:
            logger.info("%s: OK (%s file(s))", layer_name, len(files))
        else:
            logger.warning("%s: No files found in %s", layer_name, target_dir)
            all_valid = False
    
    return all_valid


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_size(bytes_count: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f} GB"


def create_bundle_archive(models_dir: Path, output_archive_path: Path, include_manifest: bool = True):
    import tarfile
    logger.info("Creating compressed bundle archive at %s...", output_archive_path)
    try:
        with tarfile.open(output_archive_path, "w:gz") as tar:
            for item in sorted(models_dir.rglob("*")):
                if not item.is_file():
                    continue
                if "archives" in item.parts:
                    continue
                if not include_manifest and item.name == "manifest.json":
                    continue
                if item == output_archive_path:
                    continue
                
                # Add to tar under relative name
                rel_name = item.relative_to(models_dir)
                tar.add(item, arcname=str(rel_name))
        logger.info("✓ Compressed archive created successfully (%s)", _format_size(output_archive_path.stat().st_size))
    except Exception as e:
        logger.error("✗ Failed to create compressed archive: %s", e)
        raise


def generate_bundle_manifest(
    *,
    bundle_version: str,
    base_url: str | None = None,
    prefix: str | None = None,
    include_manifest: bool = False,
    has_archive: bool = False,
) -> Dict:
    """Generate a manifest for the bundled models (SDK download format)."""
    manifest = {
        "bundle_version": bundle_version,
        "created_at_utc": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "files": [],
    }

    for file_path in sorted(MODELS_DIR.rglob("*")):
        if not file_path.is_file():
            continue
        if not include_manifest and file_path.name == "manifest.json":
            continue
        if "archives" in file_path.parts:
            continue
        if file_path.name == "bundle.tar.gz":
            continue

        rel_path = file_path.relative_to(MODELS_DIR).as_posix()
        manifest["files"].append({
            "path": rel_path,
            "sha256": _sha256(file_path),
        })

    if base_url:
        manifest["base_url"] = base_url
        if has_archive:
            manifest["bundle_url"] = f"{base_url.rstrip('/')}/bundle.tar.gz"
            manifest["archive_url"] = f"{base_url.rstrip('/')}/bundle.tar.gz"
    if prefix:
        manifest["prefix"] = prefix

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Bundle models from layer outputs into centralized core/models/ directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be bundled without making changes",
    )
    parser.add_argument(
        "--archive-old",
        action="store_true",
        help="Archive existing models before bundling new ones",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Save bundle manifest to this file (JSON)",
    )
    parser.add_argument(
        "--bundle-version",
        type=str,
        default=None,
        help="Bundle version string for the manifest (for example, 0.1.0).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional base URL for bundle files (e.g., https://storage.googleapis.com/bucket/models).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional prefix under the base URL (e.g., bundle/2026-05-11).",
    )
    parser.add_argument(
        "--include-manifest",
        action="store_true",
        help="Include manifest.json itself in the manifest files list.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress the bundled models into a single bundle.tar.gz archive.",
    )
    
    args = parser.parse_args()
    
    logger.info("Starting model bundling process...")
    logger.info("Repo root: %s", REPO_ROOT)
    logger.info("Models directory: %s", MODELS_DIR.relative_to(REPO_ROOT))
    
    if args.dry_run:
        logger.info("[DRY RUN] No changes will be made")
    
    results = {}
    for layer_name, config in LAYER_CONFIGS.items():
        success, message = bundle_layer(
            layer_name,
            config,
            args.dry_run,
            args.archive_old,
        )
        results[layer_name] = {"success": success, "message": message}
    
    # Validate bundle
    valid = validate_bundle()
    
    # Generate and save manifest
    if args.manifest:
        if not args.bundle_version:
            raise SystemExit("--bundle-version is required when writing a manifest.")
        manifest = generate_bundle_manifest(
            bundle_version=args.bundle_version,
            base_url=args.base_url,
            prefix=args.prefix,
            include_manifest=args.include_manifest,
            has_archive=args.compress,
        )
        manifest_path = Path(args.manifest).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        try:
            logger.info(
                "Manifest saved to: %s",
                manifest_path.relative_to(REPO_ROOT),
            )
        except ValueError:
            logger.info("Manifest saved to: %s", manifest_path)
            
        # Create compressed archive if requested
        if args.compress:
            if not args.dry_run:
                archive_path = MODELS_DIR / "bundle.tar.gz"
                create_bundle_archive(MODELS_DIR, archive_path, include_manifest=True)
            else:
                logger.info("[DRY RUN] Would create compressed archive bundle.tar.gz")
    
    logger.info("\n%s", "=" * 60)
    logger.info("SUMMARY")
    logger.info("%s", "=" * 60)
    for layer_name, result in results.items():
        status = "✓" if result["success"] else "✗"
        logger.info("%s %s: %s", status, layer_name, result["message"])
    
    if valid:
        logger.info("\n✓ Bundle validation passed")
        return 0
    else:
        logger.warning("\n✗ Bundle validation failed or incomplete")
        return 1


if __name__ == "__main__":
    exit(main())
