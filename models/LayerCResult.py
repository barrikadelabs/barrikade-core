from dataclasses import dataclass
from typing import Dict, Any
from models.LayerResult import LayerResult

@dataclass
class LayerCResult(LayerResult):
    """Standardized result from Layer C (ML Classifier)"""
    
    # Detection results
    verdict: str  # "allow", "flag", "block"
    probability_score: float  # 0.0 to 1.0 - raw probability from model
    confidence_score: float  # 0.0 to 1.0 - confidence in prediction
    
    # Processing metadata
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'verdict': self.verdict,
            'probability_score': self.probability_score,
            'confidence_score': self.confidence_score,
            'processing_time_ms': self.processing_time_ms,
        }
    
    def get_risk_score(self) -> float:
        """Calculate risk score contribution (0-100)"""
        # Scale probability to risk score
        return self.probability_score * 100.0
