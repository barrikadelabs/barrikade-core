"""Export the Layer D ModernBERT fine-tune to ONNX.

Produces an ONNX-converted bundle that core/layer_d/classifier.py auto-detects
and prefers over the PT classifier. Decouples Layer D inference from the
torch runtime: ORTModelForSequenceClassification runs on CPU via onnxruntime,
which is the deployment target for the production API container.

Workflow:
    1. Train: core/layer_d/train.py produces
       core/layer_d/outputs/model/ (BF16 transformers checkpoint)
    2. Run this script: produces
       core/layer_d/outputs/onnx/ (ONNX weights + tokenizer)
    3. Bundle: scripts/bundling/bundle_models.py copies both directories to
       core/models/layer_d/ (model/ and onnx/ as sibling dirs)
    4. Upload: scripts/bundling/gcs_upload.py ships the bundled tree to GCS

Tokenizer workaround: the trained checkpoint's tokenizer_config.json carries
tokenizer_class="TokenizersBackend" (a transformers 5.x symbol). The runtime
container pins transformers 4.x for optimum compatibility, so AutoTokenizer
fails to load it. We sidestep that by copying the raw tokenizer files
verbatim -- the ONNX export does not modify them. core/layer_d/classifier.py
loads via PreTrainedTokenizerFast directly when AutoTokenizer raises this
specific error.

Usage:
    python core/layer_d/export_layer_d_onnx.py
    python core/layer_d/export_layer_d_onnx.py --src core/layer_d/outputs/model --dst core/layer_d/outputs/onnx
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main():
    parser = argparse.ArgumentParser(description="Export Layer D model to ONNX")
    parser.add_argument(
        "--src",
        default=str(PROJECT_ROOT / "core" / "layer_d" / "outputs" / "model"),
        help="Source directory with the HF model (config.json, safetensors)",
    )
    parser.add_argument(
        "--dst",
        default=str(PROJECT_ROOT / "core" / "layer_d" / "outputs" / "onnx"),
        help="Destination directory for the ONNX bundle",
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        sys.exit(f"source dir does not exist: {src}")

    print(f"Source: {src}")
    print(f"Dest:   {dst}")
    print()

    from optimum.onnxruntime import ORTModelForSequenceClassification

    print("Exporting ModernBERT to ONNX (this can take a few minutes)...")
    t0 = time.time()
    model = ORTModelForSequenceClassification.from_pretrained(str(src), export=True)
    print(f"  Export took {time.time() - t0:.1f}s")

    print(f"Saving ONNX model to {dst}...")
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst))

    print("Copying tokenizer files verbatim (works around TokenizersBackend incompatibility)...")
    # Hard-fail on missing tokenizer.json / tokenizer_config.json: a silent
    # skip would ship an incomplete bundle that still passes the loader sentinel
    # but degrades tokenization. special_tokens_map.json is optional in HF format.
    required_tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
    optional_tokenizer_files = ("special_tokens_map.json",)
    for name in required_tokenizer_files:
        src_file = src / name
        if not src_file.exists():
            sys.exit(
                f"required tokenizer file missing from source: {src_file}. "
                f"Re-run training or pull the full Layer D model bundle."
            )
        shutil.copy2(src_file, dst / name)
    for name in optional_tokenizer_files:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)
        else:
            print(f"  warning: optional tokenizer file missing: {src_file}")

    print()
    print("--- Output ---")
    for path in sorted(dst.rglob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / 1024 / 1024
            rel = path.relative_to(dst).as_posix()
            print(f"  {size_mb:>10.2f} MB  {rel}")

    src_size = _dir_size(src)
    dst_size = _dir_size(dst)
    print()
    print("--- Sizes ---")
    print(f"PT model:    {src_size / 1024 / 1024:>10.2f} MB")
    print(f"ONNX model:  {dst_size / 1024 / 1024:>10.2f} MB")


if __name__ == "__main__":
    main()
