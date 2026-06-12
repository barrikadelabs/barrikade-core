from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OutputVerificationResult:
    """Result of verifying an LLM output with the Qwen3Guard-Stream judge.

    `risk_level` is the worst per-token level observed across the response
    (Safe < Controversial < Unsafe). `category` is only set when the output
    was flagged; the per-token lists carry one entry per response token for
    diagnostics. `truncated` means the scored sequence did not cover the full
    prompt+output (prompt context dropped and/or output tail cut).
    """

    verdict: str  # "allow" | "block"
    risk_level: str  # "Safe" | "Controversial" | "Unsafe"
    category: Optional[str]
    rationale: str
    model: str
    flagged_token_index: Optional[int]
    truncated: bool
    processing_time_ms: float
    token_risk_levels: List[str] = field(default_factory=list)
    token_categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "verdict": self.verdict,
            "risk_level": self.risk_level,
            "category": self.category,
            "rationale": self.rationale,
            "model": self.model,
            "flagged_token_index": self.flagged_token_index,
            "truncated": self.truncated,
            "processing_time_ms": self.processing_time_ms,
            "token_risk_levels": self.token_risk_levels,
            "token_categories": self.token_categories,
        }

    def get_risk_score(self) -> float:
        """Calculate risk score contribution (0-100)"""
        return 100.0 if self.verdict == "block" else 0.0
