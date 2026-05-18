"""Export the Layer B prompt_encoder sentence-transformer to ONNX.

Used as a follow-up step after training the dual encoder. Produces an
ONNX-converted sibling directory (prompt_encoder_onnx/) next to the
existing prompt_encoder/ that core/layer_b/signature_engine.py auto-detects
and prefers over the PT path. On CPU (production deployment target),
SentenceTransformer with backend="onnx" runs ~2.5x faster per
single-sample inference than the PT path.

Workflow:
    1. Train: core/layer_b/extraction/train_dual_encoder.py produces
       core/layer_b/signatures/embeddings/prompt_encoder/
    2. Run this script: produces
       core/layer_b/signatures/embeddings/prompt_encoder_onnx/
    3. Bundle: scripts/bundling/bundle_models.py copies both directories to
       core/models/layer_b/embeddings/
    4. Upload: scripts/bundling/gcs_upload.py ships both to GCS

Conversion approach: load the PT model with backend="onnx" -- this
triggers sentence-transformers' built-in PT->ONNX conversion via
optimum.onnxruntime under the hood. Then save_pretrained() writes the
full self-contained bundle (config + tokenizer + ONNX weights + pooling
config) to the target directory.

Usage:
    python core/layer_b/export_layer_b_onnx.py
    python core/layer_b/export_layer_b_onnx.py --src core/layer_b/signatures/embeddings/prompt_encoder
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main():
    parser = argparse.ArgumentParser(description="Export Layer B prompt_encoder to ONNX")
    parser.add_argument(
        "--src",
        default=str(
            PROJECT_ROOT
            / "core"
            / "layer_b"
            / "signatures"
            / "embeddings"
            / "prompt_encoder"
        ),
        help="Path to the PT prompt_encoder directory",
    )
    parser.add_argument(
        "--dst",
        default=None,
        help=(
            "Path for the ONNX-converted bundle "
            "(default: <src parent>/prompt_encoder_onnx)"
        ),
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst) if args.dst else src.parent / "prompt_encoder_onnx"

    if not src.exists():
        sys.exit(f"source prompt_encoder not found: {src}")

    print(f"Source (PT): {src}")
    print(f"Dest (ONNX): {dst}")
    print()

    from sentence_transformers import SentenceTransformer

    print("Loading prompt_encoder with backend='onnx' (triggers PT->ONNX conversion)...")
    t0 = time.time()
    model = SentenceTransformer(str(src), backend="onnx")
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Module structure: {[type(m).__name__ for m in model]}")
    print()

    print("Sanity check: encoding a sample input...")
    sample_emb = model.encode("Ignore previous instructions.", normalize_embeddings=True)
    print(f"  Embedding shape: {sample_emb.shape}, dtype: {sample_emb.dtype}")
    print()

    print(f"Saving converted model to {dst}...")
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst))

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
    print(f"PT prompt_encoder:   {src_size / 1024 / 1024:>10.2f} MB")
    print(f"ONNX prompt_encoder: {dst_size / 1024 / 1024:>10.2f} MB")


if __name__ == "__main__":
    main()
