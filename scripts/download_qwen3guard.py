"""
Download a Qwen3Guard model bundle into core/models/layer_e/.

Usage:
    python scripts/download_qwen3guard.py                    # Gen judge (Layer E)
    python scripts/download_qwen3guard.py --variant stream   # Stream output-verification judge
    python scripts/download_qwen3guard.py --force
"""

import argparse
import os
import shutil
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a Qwen3Guard bundle into the Layer E model directory"
    )
    parser.add_argument(
        "--variant",
        choices=("gen", "stream"),
        default="gen",
        help="gen = Layer E judge (default); stream = output-verification judge",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the existing local bundle if it already exists",
    )
    args = parser.parse_args()

    settings = Settings()
    if args.variant == "stream":
        # Resolved directly (not via the existence-checked settings property,
        # which raises before anything has been downloaded), honouring the
        # same env override the runtime reads.
        override = os.getenv("BARRIKADA_LAYER_E_STREAM_MODEL_DIR")
        destination = Path(override) if override else settings.layer_e_stream_model_candidates[0]
        hf_id = settings.layer_e_stream_model_hf_id
    else:
        destination = Path(settings.layer_e_model_dir)
        hf_id = settings.layer_e_model_hf_id

    if destination.exists() and any(destination.iterdir()):
        if not args.force:
            print(f"Layer E bundle already exists at {destination}")
            print("Use --force to re-download it from Hugging Face.")
            return 0
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hf_id,
        local_dir=str(destination),
    )  # type: ignore
    print(f"Downloaded {hf_id} to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
