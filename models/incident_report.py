"""Standardised incident and near-miss report schema.

Captures the full context of a workload session for post-hoc audit,
compliance reporting, and near-miss analysis.  The schema follows the
policy document's specification for structured workload records.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field



# Sub-models


class InputRecord(BaseModel):
    """Provenance record for a single input processed during the session."""

    source: str = Field(..., description="Origin of the input (e.g. 'user', 'api', 'tool_output')")
    trust_level: str = Field(..., description="trusted_internal | untrusted_external | unknown")
    content_hash: str = Field(..., description="SHA-256 prefix of the input content")
    timestamp: datetime


class ToolInvocation(BaseModel):
    """Record of a tool call made during the session."""

    tool_name: str
    timestamp: datetime
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_summary: str | None = None
    provenance: str = "unknown"


class PipelineEventRecord(BaseModel):
    """A single pipeline detection event within the session."""

    event_id: str
    timestamp: datetime
    input_hash: str
    final_verdict: str
    decision_layer: str
    confidence_score: float
    processing_time_ms: float


class InterventionRecord(BaseModel):
    """Record of a session-level intervention applied by the oversight stack."""

    intervention_type: str  # Intervention enum value
    timestamp: datetime
    reason: str
    trigger: str  # What caused the intervention (e.g. "budget_exhausted", "high_drift")
    details: dict[str, Any] = Field(default_factory=dict)


class DriftEventRecord(BaseModel):
    """Record of an intent-drift measurement."""

    timestamp: datetime
    drift_score: float
    cosine_similarity: float
    risk_level: str
    proposed_action_summary: str = ""


class RiskEventRecord(BaseModel):
    """Record of a risk-budget-consuming action."""

    timestamp: datetime
    category: str
    cost: int
    description: str
    budget_remaining_after: int


# Main Report


class IncidentReport(BaseModel):
    """Full incident / near-miss report for a workload session.

    Captures everything the policy community asks for in standardised
    near-miss data: the workload identifier, declared task, delegation
    chain, permissions flow, input provenance, model versions, tools
    invoked, oversight interventions, and final outcome.

    Crucially, this captures *near misses* — a contained prompt injection,
    a failed privilege escalation — not just realised harms.
    """

    # Identifiers
    report_id: str = Field(..., description="Unique report UUID")
    workload_id: str = Field(..., description="Session ID this report covers")
    timestamp: datetime = Field(..., description="When the report was generated")

    # Task Context
    declared_task: str
    delegation_chain: list[str] = Field(default_factory=list)

    # Permissions
    permissions_initially_granted: list[str] = Field(default_factory=list)
    permissions_later_requested: list[str] = Field(default_factory=list)
    permissions_granted_during_session: list[str] = Field(default_factory=list)

    # Input Provenance
    inputs: list[InputRecord] = Field(default_factory=list)

    # Model & System Info
    model_version: str = ""
    scaffold_version: str = ""
    barrikade_version: str = ""

    # Actions
    tools_invoked: list[ToolInvocation] = Field(default_factory=list)
    external_domains_contacted: list[str] = Field(default_factory=list)

    # Oversight Stack Decisions
    pipeline_events: list[PipelineEventRecord] = Field(default_factory=list)
    interventions: list[InterventionRecord] = Field(default_factory=list)

    # Outcome
    final_outcome: str = Field(
        ...,
        description="completed | halted | escalated | paused",
    )
    is_near_miss: bool = Field(
        False,
        description=(
            "True if the session contained a threat that was detected and "
            "contained without resulting in actual harm."
        ),
    )
    near_miss_details: str | None = None

    # Drift
    max_intent_drift_score: float = 0.0
    drift_events: list[DriftEventRecord] = Field(default_factory=list)

    # Risk Budget
    risk_budget_initial: int = 0
    risk_budget_final: int = 0
    risk_events: list[RiskEventRecord] = Field(default_factory=list)

    # Session Timing
    session_started_at: datetime | None = None
    session_ended_at: datetime | None = None
    total_duration_seconds: float | None = None
