from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from models.verdicts import DecisionLayer, FinalVerdict
from models.LayerResult import LayerResult

def _serialize_result(result: Any) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return result

@dataclass
class PipelineResult:
    input_hash: str
    total_processing_time_ms: float
    
    # Layer A results
    layer_a_result: Union[Dict[str, Any], LayerResult]
    layer_a_time_ms: float

    # Layer B results
    layer_b_result: Optional[Union[Dict[str, Any], LayerResult]]
    layer_b_time_ms: Optional[float]

    # Layer C results
    layer_c_result: Optional[Union[Dict[str, Any], LayerResult]]
    layer_c_time_ms: Optional[float]

    # Layer D results
    layer_d_result: Optional[Union[Dict[str, Any], LayerResult]]
    layer_d_time_ms: Optional[float]

    # Layer E results (LLM judge)
    layer_e_result: Optional[Union[Dict[str, Any], LayerResult]]
    layer_e_time_ms: Optional[float]
    
    # Final decision (decision cascade)
    final_verdict: FinalVerdict
    decision_layer: DecisionLayer  # "A", "B", "C", "D", or "E"
    confidence_score: float  # confidence of the deciding layer

    def to_dict(self) -> Dict[str, Any]:
        #Convert to dictionary for outpput
        return {
            'input_hash': self.input_hash,
            'total_processing_time_ms': self.total_processing_time_ms,
            'layer_a_result': _serialize_result(self.layer_a_result),
            'layer_a_time_ms': self.layer_a_time_ms,
            'layer_b_result': _serialize_result(self.layer_b_result),
            'layer_b_time_ms': self.layer_b_time_ms,
            'layer_c_result': _serialize_result(self.layer_c_result),
            'layer_c_time_ms': self.layer_c_time_ms,
            'layer_d_result': _serialize_result(self.layer_d_result),
            'layer_d_time_ms': self.layer_d_time_ms,
            'layer_e_result': _serialize_result(self.layer_e_result),
            'layer_e_time_ms': self.layer_e_time_ms,
            'final_verdict': self.final_verdict.value,
            'decision_layer': self.decision_layer.value,
            'confidence_score': self.confidence_score,
        }