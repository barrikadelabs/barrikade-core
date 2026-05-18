from enum import Enum


class DecisionLayer(str, Enum):
    LAYER_A = "A"
    LAYER_B = "B"
    LAYER_C = "C"
    LAYER_D = "D"
    LAYER_E = "E"


class FinalVerdict(str, Enum):
    """Per-request verdict emitted by each pipeline layer.

    These three values are the ONLY verdicts that ``PipelineResult`` should
    ever contain.  Session-level interventions (halt, downgrade, resample,
    escalate, revoke) are expressed via the separate ``Intervention`` enum
    and are returned by ``RiskAssessment`` / ``SessionDetectResult`` — never
    by ``PipelineResult``.  Do **not** add the agentic intervention verbs
    here; doing so would break backward compatibility for every consumer of
    the stateless ``/v1/detect`` endpoint.
    """

    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


class Intervention(str, Enum):
    """Session-level interventions applied by the Risk Budget Engine.

    These are orthogonal to ``FinalVerdict``.  A single detection request
    produces a ``FinalVerdict`` from the pipeline *and* an ``Intervention``
    from the session orchestrator.  The two are never mixed.
    """

    NONE = "none"              # No session-level action required
    HALT = "halt"              # Stop execution entirely
    DOWNGRADE = "downgrade"    # Route to a less capable model
    RESAMPLE = "resample"      # Rerun decision from a clean checkpoint
    ESCALATE = "escalate"      # Require human approval before proceeding
    REVOKE = "revoke"          # Strip permissions mid-session


class InputProvenance(str, Enum):
    """Trust level of the input source, set by the calling framework."""

    TRUSTED_INTERNAL = "trusted_internal"
    UNTRUSTED_EXTERNAL = "untrusted_external"
    UNKNOWN = "unknown"


class ResampleStrategy(str, Enum):
    """Strategy for the resample-and-compare intervention.

    TARGETED_LAST  — Strip only the most recently ingested untrusted input
                     and rerun.  Cheaper and more diagnostic: if the risky
                     action disappears, you've pinpointed the injection
                     source.
    FULL_UNTRUSTED — Strip *all* untrusted-provenance content and rerun.
                     Used as a fallback when the targeted resample is
                     inconclusive.
    """

    TARGETED_LAST = "targeted_last"
    FULL_UNTRUSTED = "full_untrusted"
