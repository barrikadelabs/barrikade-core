from dataclasses import dataclass
from typing import Dict, Any, List

from models.verdicts import InputProvenance

#make pydantic class
@dataclass
class LayerAResult:
    """Standardized result from Layer A (Text Preprocessing)"""

    
    # Input/Output text
    original_text: str
    processed_text: str
    
    # Detection metadata
    flags: List[str]
    suspicious: bool
    confidence_score: float  # 0.0 to 1.0 - confidence in the detection
    
    # Processing metadata
    processing_time_ms: float
    
    # Detailed analysis data
    decode_info: Dict[str, Any]
    confusables: Dict[str, Any]
    embedded: Dict[str, Any]

    # Input provenance — set by the session orchestrator, not by Layer A
    # itself.  Defaults to UNKNOWN for backward compatibility with the
    # stateless detect() path.
    provenance: InputProvenance = InputProvenance.UNKNOWN
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'original_text': self.original_text,
            'processed_text': self.processed_text,
            'flags': self.flags,
            'suspicious': self.suspicious,
            'confidence_score': self.confidence_score,
            'processing_time_ms': self.processing_time_ms,
            'decode_info': self.decode_info,
            'confusables': self.confusables,
            'embedded': self.embedded,
            'provenance': self.provenance.value,
        }
    
    def get_verdict(self):
        """Get verdict based on flags severity"""
        if not self.suspicious:
            return 'allow'
        
        # High severity flags that should block immediately
        high_severity_flags = ['direction_override', 'embedded_encodings']
        if any(flag in self.flags for flag in high_severity_flags):
            return 'block'
        
        # Medium severity flags that should be flagged
        medium_severity_flags = ['confusable_chars']
        if any(flag in self.flags for flag in medium_severity_flags):
            return 'flag'
        
        # Low severity flags
        return 'flag'
    
    def get_risk_score(self):
        """Calculate risk score contribution (0-100)"""
        if not self.suspicious:
            return 0.0
        
        risk_weights = {
            'direction_override': 30.0,
            'embedded_encodings': 30.0,
            'confusable_chars': 15.0,
            'possible_base64': 8.0
        }
        
        total_risk = sum(risk_weights.get(flag, 5.0) for flag in self.flags)
        return min(100.0, total_risk)
