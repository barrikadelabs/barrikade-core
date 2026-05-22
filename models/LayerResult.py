"""
Base interface for layer results to ensure consistency across all layers.
All layer result classes should follow this pattern.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Dict, Any

class LayerResult(ABC, Mapping):
    """Abstract base class for layer results"""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        pass
    
    @abstractmethod
    def get_risk_score(self) -> float:
        """
        Get risk score contribution from this layer (0-100)
        Used by orchestrator for final risk aggregation
        """
        pass
    
    def __getattr__(self, name: str) -> Any:
        """Dynamic attribute access for backward compatibility and interface enforcement."""
        if name == 'verdict':
            try:
                return self.to_dict().get('verdict', 'allow')
            except Exception:
                return 'allow'
        elif name == 'processing_time_ms':
            try:
                return float(self.to_dict().get('processing_time_ms', 0.0))
            except Exception:
                return 0.0
        elif name == 'confidence_score':
            try:
                return float(self.to_dict().get('confidence_score', 1.0))
            except Exception:
                return 1.0
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: Any) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Any:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())
