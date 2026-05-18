"""Export the Layer C XGBoost classifier from a joblib artifact to ONNX.

Used as a follow-up step after training to produce the classifier.onnx
sibling that core/layer_c/classifier.py auto-detects and prefers over the
joblib path. Once the .onnx file lives alongside the joblib, the runtime
loads via onnxruntime instead of the Python xgboost/sklearn stack.

Workflow:
    1. Train Layer C: produces core/layer_c/outputs/classifier.joblib
    2. Run this script: produces core/layer_c/outputs/classifier.onnx
    3. Bundle: scripts/bundling/bundle_models.py copies both files to core/models/layer_c/
    4. Upload: scripts/bundling/gcs_upload.py ships both to GCS

Note: the exported ONNX uses onnxmltools.convert_xgboost which produces a
ZipMap probability output (list of {0: p0, 1: p1} dicts). The runtime in
core/layer_c/classifier.py:_predict_class1_probabilities unwraps this
back to a 1-D class-1 probability array.

Usage:
    python core/layer_c/export_layer_c_onnx.py
    python core/layer_c/export_layer_c_onnx.py --src core/layer_c/outputs/classifier.joblib
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Export Layer C XGBoost classifier to ONNX")
    parser.add_argument(
        "--src",
        default=str(PROJECT_ROOT / "core" / "layer_c" / "outputs" / "classifier.joblib"),
        help="Path to the joblib artifact (default: core/layer_c/outputs/classifier.joblib)",
    )
    parser.add_argument(
        "--dst",
        default=None,
        help="Output ONNX path (default: alongside src with .onnx extension)",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=768,
        help="Embedding dimension expected by the classifier (default: 768 for mpnet-base-v2)",
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst) if args.dst else src.with_suffix(".onnx")

    if not src.exists():
        sys.exit(f"source joblib not found: {src}")

    print(f"Source: {src}")
    print(f"Dest:   {dst}")
    print()

    import joblib
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    print("Loading XGBoost classifier from joblib...")
    artifact = joblib.load(src)
    model = artifact["model"]
    calibrator = artifact.get("calibrator")
    metadata = artifact.get("metadata", {})
    print(f"  Model class:   {type(model).__name__}")
    print(f"  n_estimators:  {getattr(model, 'n_estimators', '?')}")
    print(f"  max_depth:     {getattr(model, 'max_depth', '?')}")
    print(f"  Calibrator:    {type(calibrator).__name__ if calibrator else 'None'}")
    print()

    print(f"Converting to ONNX (input shape: (None, {args.input_dim}))...")
    onnx_model = onnxmltools.convert_xgboost(
        model,
        initial_types=[("embedding", FloatTensorType([None, args.input_dim]))],
    )
    print(f"  IR version:  {onnx_model.ir_version}")
    print(f"  Inputs:      {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs:     {[o.name for o in onnx_model.graph.output]}")

    print(f"\nSaving ONNX model to {dst}...")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Also emit a calibrator-only joblib that does NOT embed the XGBoost
    # model object. The runtime ONNX path loads this and skips unpickling
    # the full classifier.joblib (which would otherwise require xgboost to
    # be installed in production for the unpickle to construct
    # XGBClassifier instances).
    calibrator_path = dst.parent / "calibrator.joblib"
    calibrator_artifact = {"calibrator": calibrator, "metadata": metadata}
    joblib.dump(calibrator_artifact, calibrator_path)
    print(f"Saved calibrator-only artifact to {calibrator_path}")

    src_size = src.stat().st_size
    dst_size = dst.stat().st_size
    cal_size = calibrator_path.stat().st_size
    print()
    print("--- Result ---")
    print(f"joblib size:     {src_size / 1024 / 1024:>8.2f} MB  (classifier.joblib, includes xgboost model)")
    print(f"ONNX size:       {dst_size / 1024 / 1024:>8.2f} MB  (classifier.onnx)")
    print(f"Calibrator size: {cal_size / 1024 / 1024:>8.2f} MB  (calibrator.joblib, no xgboost dep)")


if __name__ == "__main__":
    main()
