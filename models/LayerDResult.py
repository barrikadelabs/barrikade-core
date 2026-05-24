from dataclasses import dataclass
from typing import Dict, Any
from models.LayerResult import LayerResult

@dataclass
class LayerDResult(LayerResult):
    """Standardized result from Layer D (ModernBERT classifier)."""

    verdict: str
    probability_score: float
    confidence_score: float
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "probability_score": self.probability_score,
            "confidence_score": self.confidence_score,
            "processing_time_ms": self.processing_time_ms,
        }

    def get_risk_score(self) -> float:
        """Calculate risk score contribution (0-100)"""
        return self.probability_score * 100.0
