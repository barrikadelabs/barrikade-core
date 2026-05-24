from dataclasses import dataclass, field
from typing import List, Dict, Any
from .SignatureMatch import SignatureMatch
from models.LayerResult import LayerResult

@dataclass
class LayerBResult(LayerResult):
    """Standardized result from Layer B (Signature Detection)"""
    
    # Detection results
    matches: List[SignatureMatch]
    verdict: str  # "allow", "flag", "block"
    confidence_score: float  # 0.0 to 1.0 - confidence in detection
    
    # Processing metadata
    processing_time_ms: float
    input_hash: str

    # Optional telemetry fields for calibration and analysis
    attack_similarity: float = 0.0
    benign_similarity: float = 0.0
    contrastive_margin: float = 0.0

    # Allow-listing metadata (used for early termination / skipping later layers)
    allowlisted: bool = False
    allowlist_rules: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'input_hash': self.input_hash,
            'processing_time_ms': self.processing_time_ms,
            'matches': [
                {
                    'rule_id': match.rule_id,
                    'severity': match.severity.value,
                    'pattern': match.pattern,
                    'matched_text': match.matched_text,
                    'start_pos': match.start_pos,
                    'end_pos': match.end_pos,
                    'rule_description': match.rule_description,
                    'tags': match.tags,
                    'confidence': match.confidence
                }
                for match in self.matches
            ],
            'verdict': self.verdict,
            'confidence_score': self.confidence_score,
            'attack_similarity': self.attack_similarity,
            'benign_similarity': self.benign_similarity,
            'contrastive_margin': self.contrastive_margin,
            'allowlisted': self.allowlisted,
            'allowlist_rules': list(self.allowlist_rules),
        }

    def get_risk_score(self) -> float:
        """Calculate risk score contribution (0-100)."""
        if self.allowlisted:
            return 0.0
        if not self.matches:
            return 0.0

        # Any Layer B match is a malicious-indicator hit.
        base_risk = 50.0
        match_count_bonus = min(20.0, len(self.matches) * 5.0)
        return min(100.0, base_risk + match_count_bonus)