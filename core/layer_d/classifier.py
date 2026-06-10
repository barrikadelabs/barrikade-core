import logging
import time
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast

from models.LayerDResult import LayerDResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Thresholds:
    low: float = 0.05
    high: float = 0.95

    def validate(self):
        if not (0.0 <= self.low <= 1.0 and 0.0 <= self.high <= 1.0):
            raise ValueError("Thresholds must be within [0,1]")
        if self.low >= self.high:
            raise ValueError("Expected low < high")


class LayerDClassifier:
    @staticmethod
    def _load_tokenizer(model_dir):
        try:
            return AutoTokenizer.from_pretrained(model_dir)  # nosec B615
        except ValueError as exc:
            if "Tokenizer class" not in str(exc):
                raise

            tokenizer_json = Path(model_dir) / "tokenizer.json"
            if not tokenizer_json.exists():
                raise

            # Some local model exports carry unsupported tokenizer_class metadata.
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))
            special_tokens_map = Path(model_dir) / "special_tokens_map.json"
            if special_tokens_map.exists():
                with special_tokens_map.open("r", encoding="utf-8") as handle:
                    token_map = json.load(handle)
                for key, value in token_map.items():
                    token_value = value.get("content") if isinstance(value, dict) else value
                    if isinstance(token_value, str):
                        setattr(tokenizer, key, token_value)

            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif tokenizer.unk_token is not None:
                    tokenizer.pad_token = tokenizer.unk_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # nosec B105
            return tokenizer

    @staticmethod
    def _is_onnx_classifier_dir_ready(model_dir: Path) -> bool:
        # tokenizer_config.json is required: it triggers _load_tokenizer's
        # PreTrainedTokenizerFast fallback (the TokenizersBackend workaround).
        required_files = [
            model_dir / "config.json",
            model_dir / "model.onnx",
            model_dir / "tokenizer.json",
            model_dir / "tokenizer_config.json",
        ]
        return all(path.exists() for path in required_files)

    def _load_backend(self, model_dir):
        # Prefer ONNX when a complete sibling bundle is next to the PT model dir.
        # Decouples Layer D inference from torch (CPU via onnxruntime).
        onnx_dir = Path(model_dir).parent / "onnx"
        if onnx_dir.exists() and self._is_onnx_classifier_dir_ready(onnx_dir):
            log.info("Loading ONNX Layer D classifier: %s", onnx_dir)
            from optimum.onnxruntime import ORTModelForSequenceClassification
            tokenizer = self._load_tokenizer(str(onnx_dir))
            model = ORTModelForSequenceClassification.from_pretrained(
                str(onnx_dir),
                provider="CPUExecutionProvider",
            )
            return tokenizer, model, "cpu", True

        log.info("Loading PT Layer D classifier: %s", model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = self._load_tokenizer(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)  # nosec B615
        model.to(device)
        model.eval()
        return tokenizer, model, device, False

    def __init__(self, model_dir, low=0.05, high=0.95, max_length=512, ):
        self.max_length = max_length
        self.tokenizer, self.model, self.device, self._is_onnx = self._load_backend(model_dir)

        self.thresholds = Thresholds(low=low, high=high)
        self.thresholds.validate()

    def _score_batch(self, texts):
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]

        return probs.float().detach().cpu().numpy()

    def predict(self, input_text):
        start_time = time.time()
        probability_score = float(self._score_batch([input_text])[0])

        if probability_score < self.thresholds.low:
            verdict = "allow"
            confidence_score = 1.0 - probability_score
        elif probability_score < self.thresholds.high:
            verdict = "flag"
            confidence_score = 0.5
        else:
            verdict = "block"
            confidence_score = probability_score

        processing_time_ms = (time.time() - start_time) * 1000.0

        return LayerDResult(
            verdict=verdict,
            probability_score=probability_score,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
        )

    def predict_batch(self, texts):
        return self._score_batch(texts)
