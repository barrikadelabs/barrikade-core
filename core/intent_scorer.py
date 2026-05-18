"""Intent Deviation Scorer for longitudinal drift detection.

At session start the user's declared task is embedded.  At each subsequent
step the scorer computes a similarity delta between the current proposed
action and that original intent vector, producing a drift score that the
Risk Budget Engine and session orchestrator use for escalation decisions.

Uses a dedicated general-purpose embedding model (``all-MiniLM-L6-v2`` by
default) rather than Layer B's fine-tuned attack/benign discriminator.
The fine-tuned model's embedding space is distorted to maximise the gap
between attack and benign content - that distortion would produce
misleading drift scores for arbitrary task descriptions.

The model is loaded **lazily** (on first use) so it does not affect the
stateless ``detect()`` hot path.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from core.session_settings import SessionSettings

log = logging.getLogger(__name__)


class DriftLevel(str, Enum):
    """Risk classification of an intent-drift score."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of a single intent-drift measurement.

    Attributes:
        drift_score: 0.0 (identical to intent) to 1.0 (orthogonal).
        cosine_similarity: Raw cosine similarity (higher = more similar).
        risk_level: Classified drift level.
        proposed_action_vector: Embedding of the proposed action, kept
            for downstream analysis or logging.
    """

    drift_score: float
    cosine_similarity: float
    risk_level: DriftLevel
    proposed_action_vector: np.ndarray

    def to_dict(self) -> dict:
        return {
            "drift_score": round(self.drift_score, 4),
            "cosine_similarity": round(self.cosine_similarity, 4),
            "risk_level": self.risk_level.value,
        }


class IntentDeviationScorer:
    """Computes intent drift between a session's declared intent and
    subsequent proposed actions.

    The embedding model is loaded lazily on first call to ``embed_intent``
    or ``compute_drift``.
    """

    def __init__(
        self,
        settings: SessionSettings | None = None,
        model: "SentenceTransformer | None" = None,
    ) -> None:
    
        self._settings = settings or SessionSettings()
        self._model: SentenceTransformer | None = model
        self._model_name = self._settings.intent_embedding_model

    #lazy model loading 

    def _ensure_model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info(
                "Loading intent embedding model: %s (lazy init)",
                self._model_name,
            )
            self._model = SentenceTransformer(self._model_name)
        return self._model

    #public API

    def embed_intent(self, declared_intent: str) -> np.ndarray:
        """Embed the declared task intent into a vector.

        Called once at session creation; the resulting vector is stored in
        ``WorkloadSession.intent_vector``.
        """
        model = self._ensure_model()
        vec = model.encode(
            [declared_intent],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vec[0].astype(np.float32)

    def compute_drift(
        self,
        intent_vector: np.ndarray,
        proposed_action: str,
    ) -> DriftResult:
        """Compute the drift between the original intent and a proposed action.

        Args:
            intent_vector: The session's intent embedding (from
                ``embed_intent``).
            proposed_action: Free-text description of the current proposed
                action or the input text being screened.

        Returns:
            DriftResult with the drift score and risk classification.
        """
        model = self._ensure_model()
        action_vec = model.encode(
            [proposed_action],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0].astype(np.float32)

        # Both vectors are L2-normalised, so dot product = cosine sim
        cosine_sim = float(np.dot(intent_vector, action_vec))
        # Clamp to [0, 1] to handle floating-point edge cases
        cosine_sim = max(0.0, min(1.0, cosine_sim))
        drift = 1.0 - cosine_sim

        risk_level = self._classify_drift(drift)

        return DriftResult(
            drift_score=drift,
            cosine_similarity=cosine_sim,
            risk_level=risk_level,
            proposed_action_vector=action_vec,
        )

    def _classify_drift(self, drift_score: float) -> DriftLevel:
        if drift_score >= self._settings.intent_drift_block_threshold:
            return DriftLevel.CRITICAL
        if drift_score >= self._settings.intent_drift_warn_threshold:
            return DriftLevel.HIGH
        if drift_score >= self._settings.intent_drift_warn_threshold * 0.6:
            return DriftLevel.MODERATE
        return DriftLevel.LOW
