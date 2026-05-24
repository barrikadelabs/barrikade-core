"""Tests for scripts/bundling/bundle_models.py.

Pins the per-layer required_patterns so a future regression that re-broadens
them can't silently re-ship artifacts the runtime never loads. Each layer
has one test asserting the expected file set against a synthesized source
directory.

Layer C: had ~160 MB of dead-weight joblibs (pca_reducer, rf_model,
tf_idf_*, feature_selector, meta_features) left over from earlier classifier
experiments — none consumed at runtime by core/layer_c/classifier.py.

Layer B: had ~138 MB of artifacts the runtime ignores — the signature_encoder
sentence-transformer (only prompt_encoder is loaded by
core/layer_b/signature_engine.py) and the extracted/*.yar pattern files
(no .yar references in any core/*.py).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bundling.bundle_models import LAYER_CONFIGS, get_model_files


def test_layer_c_pattern_excludes_dead_weight(tmp_path):
    """Layer C bundling must match only classifier.joblib + classifier.onnx
    + calibrator.joblib + encoder_onnx/ (current + pinned releases), not
    other joblibs from earlier experimental pipelines."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    # Files that must be bundled
    (outputs / "classifier.joblib").write_bytes(b"current_joblib")
    (outputs / "classifier.onnx").write_bytes(b"current_onnx")
    (outputs / "calibrator.joblib").write_bytes(b"calibrator_only")
    release_dir = outputs / "releases" / "v0.1"
    release_dir.mkdir(parents=True)
    (release_dir / "classifier.joblib").write_bytes(b"v0.1_joblib")

    # ONNX-converted encoder bundle (produced by
    # core/layer_c/export_layer_c_encoder_onnx.py), runtime-loaded by
    # core/layer_c/classifier.py which prefers it over the HF Hub PT model.
    encoder_onnx = outputs / "encoder_onnx"
    encoder_onnx.mkdir()
    (encoder_onnx / "config.json").write_text("{}")
    (encoder_onnx / "modules.json").write_text("[]")
    (encoder_onnx / "tokenizer.json").write_text("{}")
    onnx_subdir = encoder_onnx / "onnx"
    onnx_subdir.mkdir()
    (onnx_subdir / "model.onnx").write_bytes(b"onnx_weights")

    # Vestigial files that must NOT be bundled
    for name in (
        "pca_reducer.joblib",
        "rf_model.joblib",
        "tf_idf_logreg.joblib",
        "tf_idf_vectorizer.joblib",
        "feature_selector.joblib",
        "meta_features.joblib",
    ):
        (outputs / name).write_bytes(b"dead")

    patterns = LAYER_CONFIGS["layer_c"].required_patterns
    matched = get_model_files(outputs, patterns)
    matched_relative = {p.relative_to(outputs).as_posix() for p in matched if p.is_file()}

    assert matched_relative == {
        "classifier.joblib",
        "classifier.onnx",
        "calibrator.joblib",
        "releases/v0.1/classifier.joblib",
        "encoder_onnx/config.json",
        "encoder_onnx/modules.json",
        "encoder_onnx/tokenizer.json",
        "encoder_onnx/onnx/model.onnx",
    }


def test_layer_d_pattern_includes_onnx_bundle(tmp_path):
    """Layer D bundling must ship both the PT model/ directory and the
    onnx/ sibling produced by core/layer_d/export_layer_d_onnx.py. The runtime
    classifier auto-detects onnx/ and prefers it over PT."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    # PT model dir
    model_dir = outputs / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"pt_weights")
    (model_dir / "tokenizer.json").write_text("{}")
    (model_dir / "tokenizer_config.json").write_text("{}")

    # ONNX sibling bundle
    onnx_dir = outputs / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "config.json").write_text("{}")
    (onnx_dir / "model.onnx").write_bytes(b"onnx_weights")
    (onnx_dir / "tokenizer.json").write_text("{}")
    (onnx_dir / "tokenizer_config.json").write_text("{}")

    patterns = LAYER_CONFIGS["layer_d"].required_patterns
    matched = get_model_files(outputs, patterns)
    matched_relative = {p.relative_to(outputs).as_posix() for p in matched if p.is_file()}

    # Both PT and ONNX bundles must be included
    assert "model/config.json" in matched_relative
    assert "model/model.safetensors" in matched_relative
    assert "model/tokenizer.json" in matched_relative
    assert "onnx/config.json" in matched_relative
    assert "onnx/model.onnx" in matched_relative
    assert "onnx/tokenizer.json" in matched_relative


def test_layer_b_pattern_excludes_unused_artifacts(tmp_path):
    """Layer B bundling must ship only files the runtime loads:
    centroids/FAISS indices, the small JSON metadata, and prompt_encoder/.
    It must NOT ship signature_encoder/ (only used by the offline
    extract_signature_patterns.py) or extracted/*.yar (not referenced by
    any runtime Python code)."""
    signatures = tmp_path / "signatures"
    signatures.mkdir()

    embeddings = signatures / "embeddings"
    embeddings.mkdir()

    # Files that must be bundled (loaded by core/layer_b/signature_engine.py)
    (embeddings / "centroids.npy").write_bytes(b"centroids")
    (embeddings / "benign_centroids.npy").write_bytes(b"benign_centroids")
    (embeddings / "cluster_radii.json").write_text("{}")
    (embeddings / "metadata.json").write_text("{}")
    (embeddings / "faiss_index.bin").write_bytes(b"faiss")
    (embeddings / "benign_faiss_index.bin").write_bytes(b"benign_faiss")

    prompt_encoder = embeddings / "prompt_encoder"
    prompt_encoder.mkdir()
    (prompt_encoder / "config.json").write_text("{}")
    (prompt_encoder / "model.safetensors").write_bytes(b"prompt_weights")
    (prompt_encoder / "tokenizer.json").write_text("{}")

    # ONNX-converted encoder bundle (produced by core/layer_b/export_layer_b_onnx.py),
    # also runtime-loaded — signature_engine.py auto-detects this and prefers
    # it over the PT prompt_encoder when present.
    prompt_encoder_onnx = embeddings / "prompt_encoder_onnx"
    prompt_encoder_onnx.mkdir()
    (prompt_encoder_onnx / "config.json").write_text("{}")
    (prompt_encoder_onnx / "modules.json").write_text("[]")
    (prompt_encoder_onnx / "tokenizer.json").write_text("{}")
    onnx_subdir = prompt_encoder_onnx / "onnx"
    onnx_subdir.mkdir()
    (onnx_subdir / "model.onnx").write_bytes(b"onnx_weights")

    # Files that must NOT be bundled
    signature_encoder = embeddings / "signature_encoder"
    signature_encoder.mkdir()
    (signature_encoder / "config.json").write_text("{}")
    (signature_encoder / "model.safetensors").write_bytes(b"sig_weights")
    (signature_encoder / "tokenizer.json").write_text("{}")

    extracted = signatures / "extracted"
    extracted.mkdir()
    (extracted / "malicious_block_high_signatures.yar").write_text("rule x {}")
    (extracted / "safe_allow_signatures.yar").write_text("rule y {}")

    patterns = LAYER_CONFIGS["layer_b"].required_patterns
    matched = get_model_files(signatures, patterns)
    matched_relative = {p.relative_to(signatures).as_posix() for p in matched if p.is_file()}

    assert matched_relative == {
        "embeddings/centroids.npy",
        "embeddings/benign_centroids.npy",
        "embeddings/cluster_radii.json",
        "embeddings/metadata.json",
        "embeddings/faiss_index.bin",
        "embeddings/benign_faiss_index.bin",
        "embeddings/prompt_encoder/config.json",
        "embeddings/prompt_encoder/model.safetensors",
        "embeddings/prompt_encoder/tokenizer.json",
        "embeddings/prompt_encoder_onnx/config.json",
        "embeddings/prompt_encoder_onnx/modules.json",
        "embeddings/prompt_encoder_onnx/tokenizer.json",
        "embeddings/prompt_encoder_onnx/onnx/model.onnx",
    }
