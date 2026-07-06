import core.onnx_patch
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from models.LayerCResult import LayerCResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Thresholds:
    low: float = 0.35
    high: float = 0.85

    def validate(self):
        if not (0.0 <= self.low <= 1.0 and 0.0 <= self.high <= 1.0):
            raise ValueError("Thresholds must be within [0,1]")
        if self.low >= self.high:
            raise ValueError("Expected low < high")


class Classifier:
    @staticmethod
    def _is_onnx_encoder_dir_ready(model_dir: Path) -> bool:
        required_files = [
            model_dir / "config.json",
            model_dir / "modules.json",
            model_dir / "tokenizer.json",
            model_dir / "onnx" / "model.onnx",
        ]
        return all(path.exists() for path in required_files)

    def _load_encoder(self, model_path: Path, embedding_model: str):
        # Prefer the ONNX encoder bundle when present (2-3x faster CPU inference).
        # Walk parents in case model_path resolves under releases/<v>/ — the
        # bundle still lives at the layer_c root, not inside the release dir.
        model_path = Path(model_path)
        onnx_dir = None
        for parent_dir in (model_path.parent, *list(model_path.parent.parents)[:2]):
            candidate = parent_dir / "encoder_onnx"
            if candidate.exists() and self._is_onnx_encoder_dir_ready(candidate):
                onnx_dir = candidate
                break

        if onnx_dir is not None:
            log.info("Loading ONNX Layer C encoder: %s", onnx_dir)
            return SentenceTransformer(
                str(onnx_dir),
                backend="onnx",
                model_kwargs={"providers": ["CPUExecutionProvider"]},
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading PT Layer C encoder: %s (device=%s)", embedding_model, device)
        return SentenceTransformer(embedding_model, device=device)

    def __init__(self, model_path, embedding_model="all-mpnet-base-v2", low=0.35, high=0.85, ):
        self.encoder = self._load_encoder(model_path, embedding_model)

        # Prefer the ONNX classifier sibling if it exists; fall back to the
        # sklearn XGBoost in the joblib otherwise. The ONNX path is structured
        # so that it does NOT need to unpickle classifier.joblib (which
        # embeds the XGBClassifier object and would still require xgboost to
        # be installed) — it loads the calibrator from a separate
        # calibrator.joblib artifact written alongside classifier.onnx by
        # core/layer_c/export_layer_c_onnx.py. Once classifier.joblib is no longer
        # bundled, xgboost can drop from the runtime image entirely.
        onnx_path = Path(model_path).with_suffix(".onnx")
        calibrator_path = Path(model_path).parent / "calibrator.joblib"
        if onnx_path.exists():
            # ONNX path: avoid unpickling the full classifier.joblib if a
            # standalone calibrator artifact is present.
            if calibrator_path.exists():
                cal_artifact = joblib.load(calibrator_path)
                self.calibrator = cal_artifact.get("calibrator")
            else:
                # Backward compat: pre-split bundles still have everything
                # in the full joblib. Loading it here requires xgboost; that
                # constraint goes away once calibrator.joblib is the norm.
                artifact = joblib.load(model_path)
                self.calibrator = artifact.get("calibrator")
            import onnxruntime as ort
            self._onnx_session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self._onnx_input_name = self._onnx_session.get_inputs()[0].name
            self.model = None
        else:
            # sklearn fallback path: load model + calibrator from the full
            # joblib as before.
            artifact = joblib.load(model_path)
            self.calibrator = artifact.get("calibrator")
            self._onnx_session = None
            self.model = artifact.get("model")
            if self.model is None or not hasattr(self.model, "predict_proba"):
                raise ValueError(
                    "Layer C model artifact does not contain a valid "
                    "predict_proba model and no classifier.onnx was found "
                    "alongside the joblib"
                )

        self.thresholds = Thresholds(low=low, high=high)
        self.thresholds.validate()

    def _predict_class1_probabilities(self, embeddings):
        """Return a 1-D array of class-1 probabilities for each row of
        embeddings. Selects between the ONNX and sklearn backends based on
        which one is loaded."""
        if self._onnx_session is not None:
            # onnxmltools.convert_xgboost emits a ZipMap output, so the
            # second output is a list of {0: p0, 1: p1} dicts rather than
            # a numpy array. Unwrap to extract only the class-1 column.
            embeddings = np.asarray(embeddings, dtype=np.float32)
            outputs = self._onnx_session.run(
                None, {self._onnx_input_name: embeddings}
            )
            zipmap = outputs[1]
            return np.array([d[1] for d in zipmap], dtype=np.float32) # type: ignore
        return self.model.predict_proba(embeddings)[:, 1] # type: ignore

    def predict(self, input_text):
        start_time = time.time()

        emb = self.encoder.encode([input_text], normalize_embeddings=True)
        probability_score = float(self._predict_class1_probabilities(emb)[0])
        if self.calibrator is not None:
            probability_score = float(self.calibrator.predict(np.array([probability_score]))[0])

        if probability_score < self.thresholds.low:
            verdict = "allow"
        elif probability_score < self.thresholds.high:
            verdict = "flag"
        else:
            verdict = "block"

        # Confidence: distance from the decision boundary
        if verdict == "allow":
            confidence_score = 1.0 - probability_score
        elif verdict == "block":
            confidence_score = probability_score
        else:
            # Middle band = uncertain
            confidence_score = 0.5

        processing_time_ms = (time.time() - start_time) * 1000.0

        return LayerCResult(
            verdict=verdict,
            probability_score=probability_score,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
        )

    def predict_dict(self, input_text):
        """Something to get a simple dict output for API responses."""
        res = self.predict(input_text)
        return {"score": res.probability_score, "decision": res.verdict}

    def predict_batch(self, texts):
        """Return raw probability scores for a batch of texts."""
        embs = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        probs = self._predict_class1_probabilities(embs)
        if self.calibrator is not None:
            probs = self.calibrator.predict(probs)
        return probs