"""
Validate that bundled models are present and loadable.

This script checks:
1. All required files exist in core/models/
2. Models can be loaded without errors
3. Model directory structure is correct

Usage:
    python scripts/bundling/validate_models.py [--verbose]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORE_DIR = REPO_ROOT / "core"
MODELS_DIR = CORE_DIR / "models"


def validate_layer_b() -> Tuple[bool, List[str]]:
    """Validate Layer B (Signature Engine) models."""
    errors = []
    warnings = []
    target_dir = MODELS_DIR / "layer_b" / "embeddings"
    
    if not target_dir.exists():
        errors.append(f"Directory does not exist: {target_dir}")
        return False, errors

    required_files = [
        target_dir / "centroids.npy",
        target_dir / "metadata.json",
        target_dir / "cluster_radii.json",
        target_dir / "prompt_encoder" / "config.json",
        target_dir / "prompt_encoder" / "modules.json",
        target_dir / "prompt_encoder" / "tokenizer.json",
        target_dir / "signature_encoder" / "config.json",
        target_dir / "signature_encoder" / "modules.json",
        target_dir / "signature_encoder" / "tokenizer.json",
    ]
    missing_required = [str(path.relative_to(MODELS_DIR)) for path in required_files if not path.exists()]
    if missing_required:
        errors.extend(f"Missing required Layer B file: {path}" for path in missing_required)

    prompt_weight_files = [
        target_dir / "prompt_encoder" / "model.safetensors",
        target_dir / "prompt_encoder" / "pytorch_model.bin",
    ]
    signature_weight_files = [
        target_dir / "signature_encoder" / "model.safetensors",
        target_dir / "signature_encoder" / "pytorch_model.bin",
    ]
    has_prompt_weights = any(path.exists() for path in prompt_weight_files)
    has_signature_weights = any(path.exists() for path in signature_weight_files)
    if not has_prompt_weights:
        errors.append(
            "Missing Layer B prompt encoder weights under core/models/layer_b/embeddings/prompt_encoder/"
        )
    if not has_signature_weights:
        errors.append(
            "Missing Layer B signature encoder weights under core/models/layer_b/embeddings/signature_encoder/"
        )

    has_attack_index = (target_dir / "faiss_index.bin").exists()
    has_benign_index = (target_dir / "benign_faiss_index.bin").exists()
    if not has_attack_index:
        warnings.append("Missing Layer B FAISS attack index (faiss_index.bin); sklearn fallback is still possible")
    if not has_benign_index:
        warnings.append("Missing Layer B FAISS benign index (benign_faiss_index.bin); sklearn fallback is still possible")

    logger.info("Layer B:")
    logger.info(f"  ✓ Embeddings directory: {target_dir.exists()}")
    logger.info(f"  ✓ Prompt encoder weights: {has_prompt_weights}")
    logger.info(f"  ✓ Signature encoder weights: {has_signature_weights}")
    logger.info(f"  ✓ Attack centroids: {(target_dir / 'centroids.npy').exists()}")
    logger.info(f"  ✓ Metadata: {(target_dir / 'metadata.json').exists()}")
    logger.info(f"  ✓ FAISS attack index: {has_attack_index}")
    logger.info(f"  ✓ FAISS benign index: {has_benign_index}")

    try:
        from sentence_transformers import SentenceTransformer
        if has_prompt_weights:
            SentenceTransformer(str(target_dir / "prompt_encoder"), device="cpu")
            logger.info("  ✓ Prompt encoder loads successfully")
        if has_signature_weights:
            SentenceTransformer(str(target_dir / "signature_encoder"), device="cpu")
            logger.info("  ✓ Signature encoder loads successfully")
    except ImportError:
        logger.debug("  ⊘ sentence-transformers not installed, skipping encoder load test")
    except Exception as e:
        errors.append(f"Failed to load Layer B sentence-transformer bundle: {e}")

    try:
        import faiss
        if has_attack_index:
            faiss.read_index(str(target_dir / "faiss_index.bin"))
            logger.info("  ✓ FAISS attack index loads successfully")
        if has_benign_index:
            faiss.read_index(str(target_dir / "benign_faiss_index.bin"))
            logger.info("  ✓ FAISS benign index loads successfully")
    except ImportError:
        logger.debug("  ⊘ faiss not installed, skipping load test")
    except Exception as e:
        errors.append(f"Failed to load FAISS index: {e}")

    for warning in warnings:
        logger.warning(f"  ⚠ {warning}")

    return len(errors) == 0, errors


def validate_layer_c() -> Tuple[bool, List[str]]:
    """Validate Layer C (ML Classifier) models."""
    errors = []
    target_dir = MODELS_DIR / "layer_c"
    
    if not target_dir.exists():
        errors.append(f"Directory does not exist: {target_dir}")
        return False, errors
    
    # Check for joblib files
    joblib_files = list(target_dir.glob("**/*.joblib"))
    
    if not joblib_files:
        errors.append("No .joblib files found")
        return False, errors
    
    logger.info("Layer C:")
    logger.info(f"  ✓ Joblib files: {len(joblib_files)}")
    
    # Try to load a sample joblib file
    try:
        import joblib
        test_file = joblib_files[0]
        model = joblib.load(test_file)
        logger.info(f"  ✓ Joblib model loads successfully ({test_file.name})")
    except ImportError:
        logger.debug("  ⊘ joblib not installed, skipping load test")
    except Exception as e:
        errors.append(f"Failed to load joblib model: {e}")
    
    return len(errors) == 0, errors


def validate_layer_d() -> Tuple[bool, List[str]]:
    """Validate Layer D (ModernBERT) models."""
    errors = []
    target_dir = MODELS_DIR / "layer_d"
    
    if not target_dir.exists():
        errors.append(f"Directory does not exist: {target_dir}")
        return False, errors
    
    # Check for Hugging Face model structure
    # Models can be in model/ subdirectory (from bundling) or at root
    model_dir = target_dir / "model"
    config_file = target_dir / "config.json" if (target_dir / "config.json").exists() else target_dir / "model" / "config.json"
    tokenizer_file = target_dir / "tokenizer.json" if (target_dir / "tokenizer.json").exists() else target_dir / "model" / "tokenizer.json"
    
    has_model_dir = model_dir.exists()
    has_config = config_file.exists()
    has_tokenizer = tokenizer_file.exists()
    
    logger.info("Layer D:")
    logger.info(f"  ✓ Model directory: {has_model_dir}")
    logger.info(f"  ✓ Config file: {has_config}")
    logger.info(f"  ✓ Tokenizer file: {has_tokenizer}")
    
    if not has_config:
        errors.append("Missing config.json")
    if not has_tokenizer and not has_config:
        errors.append("Missing both config.json and tokenizer.json")
    
    # Try to load model if transformers is available
    try:
        from transformers import AutoModel, AutoTokenizer
        
        if has_config and has_model_dir:
            try:
                AutoModel.from_pretrained(str(model_dir))  # nosec B615
                logger.info(f"  ✓ Transformers model loads successfully")
            except Exception as e:
                logger.warning(f"  ⚠ Warning loading model: {e}")
    except ImportError:
        logger.debug("  ⊘ transformers not installed, skipping load test")
    
    return len(errors) == 0, errors


def validate_layer_e() -> Tuple[bool, List[str]]:
    """Validate Layer E (LLM Judge) models."""
    errors = []
    target_dir = MODELS_DIR / "layer_e"
    
    if not target_dir.exists():
        errors.append(f"Directory does not exist: {target_dir}")
        return False, errors
    
    qwen3guard_dir = target_dir / "qwen3guard-barrikade"
    model_dir = qwen3guard_dir
    has_model_dir = model_dir.exists()
    
    # Check for config files
    config_files = list(target_dir.glob("**/*.json"))

    logger.info("Layer E:")
    logger.info(f"  ✓ Qwen3Guard directory: {qwen3guard_dir.exists()}")
    logger.info(f"  ✓ Config files: {len(config_files)}")

    if not has_model_dir and not config_files:
        errors.append("Qwen3Guard bundle not found")
    
    # Try to load Layer E model if available
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if model_dir.exists():
            try:
                AutoModelForCausalLM.from_pretrained(str(model_dir))  # nosec B615
                logger.info(f"  ✓ Layer E model loads successfully")
            except Exception as e:
                logger.warning(f"  ⚠ Warning loading Layer E model: {e}")
    except ImportError:
        logger.debug("  ⊘ transformers not installed, skipping load test")
    
    return len(errors) == 0, errors


def check_archive_structure() -> Tuple[bool, List[str]]:
    """Check that archive directories exist."""
    logger.info("\nArchive structure:")
    for layer in ["layer_b", "layer_c", "layer_d", "layer_e"]:
        archive_dir = MODELS_DIR / layer / "archives"
        exists = archive_dir.exists()
        logger.info(f"  {layer}/archives: {exists}")
        if not exists:
            logger.warning(f"  ⚠ Archive directory missing: {archive_dir}")

    return True, []


def main():
    parser = argparse.ArgumentParser(description="Validate bundled models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Validating bundled models...")
    logger.info(f"Models directory: {MODELS_DIR.relative_to(REPO_ROOT)}\n")
    
    results = {
        "Layer B (Signatures)": validate_layer_b(),
        "Layer C (Classifier)": validate_layer_c(),
        "Layer D (ModernBERT)": validate_layer_d(),
        "Layer E (LLM Judge)": validate_layer_e(),
        "Archive structure": check_archive_structure(),
    }
    
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    all_valid = True
    for name, (success, errors) in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {name}")
        
        if errors:
            for error in errors:
                logger.error(f"  → {error}")
            all_valid = False
    
    if all_valid:
        logger.info(f"\n✓ All validations passed")
        return 0
    else:
        logger.error(f"\n✗ Some validations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
