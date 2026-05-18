# Development Guide

This guide covers developer-facing workflows for Barrikada.

## Local setup

1. Create and activate a Python 3.10+ virtual environment (3.11 recommended; matches the production container).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. The SDK auto-fetches the runtime bundle into `~/.barrikade/bundle/` on first import of `barrikade`.
   Set `BARRIKADA_SKIP_IMPORT_BUNDLE_CHECK=1` to skip. See `docs/MODEL_HOSTING.md` for details.

Manual downloads:

```bash
python scripts/bundling/gcs_download.py --bucket barrikade-bundles
python scripts/download_qwen3guard.py
```

## Model distribution

Runtime models and datasets are distributed via the `barrikade-bundles` Google Cloud Storage public bucket.

**Key points:**
- No credentials needed for download (public read access)
- The SDK auto-fetches the bundle on first import of `barrikade`
- Docker containers auto-download at startup via `docker_entrypoint.sh`
- Manual downloads via `scripts/bundling/gcs_download.py` (full bundle) and `scripts/download_qwen3guard.py` (Layer E only)

For complete documentation, see `docs/MODEL_HOSTING.md`.

## Training Layer B encoders

To train custom dual-encoder models for Layer B:

```bash
python core/layer_b/extraction/train_dual_encoder.py
```

After training, rebuild signatures:

```bash
python core/layer_b/extraction/extract_signature_patterns.py
```

## Docker workflow

See `docs/DOCKER.md` for image build, compose, health checks, and runtime environment variables.

## Quality checks

```bash
pytest -q
```

## Examples

```bash
python examples/quickstart.py
python examples/basic_detection.py
```
