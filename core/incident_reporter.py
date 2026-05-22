"""Incident Reporter — assembles structured reports from session events.

Walks the session event log and produces a full ``IncidentReport`` that
captures the complete context of a workload session for audit, compliance,
and near-miss analysis.
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from core.session import (
    SessionEvent,
    SessionEventType,
    SessionStoreBackend
)
from models.incident_report import (
    DriftEventRecord,
    IncidentReport,
    InputRecord,
    InterventionRecord,
    PipelineEventRecord,
    RiskEventRecord,
    ToolInvocation,
)

log = logging.getLogger(__name__)

# Version constants (kept here so the reporter can stamp them)
_BARRIKADE_VERSION = "0.2.0" #TODO: Update this to a more permanent settings setup


class IncidentReporter:
    """Assembles ``IncidentReport`` objects from completed sessions."""

    def __init__(self, session_store: SessionStoreBackend) -> None:
        self._store = session_store

    #public API

    def generate_report(
        self,
        session_id: str,
        model_version: str = "",
        scaffold_version: str = "",
    ) -> IncidentReport:
        """Build a full incident report for the given session.

        Args:
            session_id: The session to report on (may be active or closed).
            model_version: Optional model version string for the report.
            scaffold_version: Optional scaffold version string.

        Returns:
            IncidentReport populated from session events.
        """
        session = self._store.get_session(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} not found")

        pipeline_events: list[PipelineEventRecord] = []
        interventions: list[InterventionRecord] = []
        drift_events: list[DriftEventRecord] = []
        risk_events: list[RiskEventRecord] = []
        tools_invoked: list[ToolInvocation] = []
        inputs: list[InputRecord] = []
        permissions_requested: list[str] = []
        permissions_granted_during: list[str] = []
        near_miss_detected = False
        near_miss_reasons: list[str] = []
        max_drift = 0.0

        for event in session.events:
            self._process_event(
                event,
                pipeline_events=pipeline_events,
                interventions=interventions,
                drift_events=drift_events,
                risk_events=risk_events,
                tools_invoked=tools_invoked,
                inputs=inputs,
                permissions_requested=permissions_requested,
                permissions_granted_during=permissions_granted_during,
            )

        #Near-miss detection
        # If any pipeline event returned block or flag but the session
        # was ultimately allowed to complete, that's a near miss.
        for pe in pipeline_events:
            if pe.final_verdict in ("block", "flag"):
                near_miss_detected = True
                near_miss_reasons.append(
                    f"Pipeline {pe.final_verdict} at layer {pe.decision_layer} "
                    f"(event {pe.event_id}, confidence={pe.confidence_score:.2f})"
                )

        # Also check interventions that were triggered but didn't halt
        for iv in interventions:
            if iv.intervention_type in ("escalate", "resample", "downgrade"):
                near_miss_detected = True
                near_miss_reasons.append(
                    f"Intervention {iv.intervention_type}: {iv.reason}"
                )

        # Max drift
        if drift_events:
            max_drift = max(de.drift_score for de in drift_events)

        # Duration
        duration = None
        if session.closed_at:
            duration = (session.closed_at - session.created_at).total_seconds()

        return IncidentReport(
            report_id=uuid4().hex,
            workload_id=session.session_id,
            timestamp=datetime.now(timezone.utc),
            declared_task=session.declared_intent,
            delegation_chain=list(session.delegation_chain),
            permissions_initially_granted=list(session.permissions_granted),
            permissions_later_requested=permissions_requested,
            permissions_granted_during_session=permissions_granted_during,
            inputs=inputs,
            model_version=model_version,
            scaffold_version=scaffold_version,
            barrikade_version=_BARRIKADE_VERSION,
            tools_invoked=tools_invoked,
            external_domains_contacted=list(session.external_domains_contacted),
            pipeline_events=pipeline_events,
            interventions=interventions,
            final_outcome=session.status.value,
            is_near_miss=near_miss_detected,
            near_miss_details=(
                "; ".join(near_miss_reasons) if near_miss_reasons else None
            ),
            max_intent_drift_score=max_drift,
            drift_events=drift_events,
            risk_budget_initial=session.risk_budget_initial,
            risk_budget_final=session.risk_budget_remaining,
            risk_events=risk_events,
            session_started_at=session.created_at,
            session_ended_at=session.closed_at,
            total_duration_seconds=duration,
        )

    def export_json(self, report: IncidentReport) -> str:
        """Serialize report to JSON string."""
        return report.model_dump_json(indent=2)

    def export_dict(self, report: IncidentReport) -> dict[str, Any]:
        """Serialize report to dict (for API responses)."""
        return report.model_dump(mode="json")

    #event processing

    def _process_event(
        self,
        event: SessionEvent,
        *,
        pipeline_events: list[PipelineEventRecord],
        interventions: list[InterventionRecord],
        drift_events: list[DriftEventRecord],
        risk_events: list[RiskEventRecord],
        tools_invoked: list[ToolInvocation],
        inputs: list[InputRecord],
        permissions_requested: list[str],
        permissions_granted_during: list[str],
    ) -> None:
        """Route a single session event to the appropriate report list."""
        data = event.data

        if event.event_type == SessionEventType.PIPELINE_RESULT:
            pr = event.pipeline_result or {}
            pipeline_events.append(PipelineEventRecord(
                event_id=event.event_id,
                timestamp=event.timestamp,
                input_hash=pr.get("input_hash", ""),
                final_verdict=pr.get("final_verdict", "unknown"),
                decision_layer=pr.get("decision_layer", ""),
                confidence_score=pr.get("confidence_score", 0.0),
                processing_time_ms=pr.get("total_processing_time_ms", 0.0),
            ))
            # Also record the input provenance
            inputs.append(InputRecord(
                source=data.get("source", "pipeline"),
                trust_level=event.provenance.value,
                content_hash=pr.get("input_hash", ""),
                timestamp=event.timestamp,
            ))

        elif event.event_type == SessionEventType.INTERVENTION:
            interventions.append(InterventionRecord(
                intervention_type=data.get("intervention", "unknown"),
                timestamp=event.timestamp,
                reason=data.get("reason", ""),
                trigger=data.get("trigger", "unknown"),
                details=data.get("details", {}),
            ))

        elif event.event_type == SessionEventType.DRIFT_CHECK:
            drift_events.append(DriftEventRecord(
                timestamp=event.timestamp,
                drift_score=data.get("drift_score", 0.0),
                cosine_similarity=data.get("cosine_similarity", 0.0),
                risk_level=data.get("risk_level", "low"),
                proposed_action_summary=data.get("proposed_action_summary", ""),
            ))

        elif event.event_type == SessionEventType.RISK_BUDGET_DEDUCTION:
            categories = data.get("categories", [])
            cost = data.get("total_cost", 0)
            remaining = data.get("budget_remaining", 0)
            for cat in categories:
                risk_events.append(RiskEventRecord(
                    timestamp=event.timestamp,
                    category=cat,
                    cost=cost // max(len(categories), 1),
                    description=data.get("description", f"{cat} risk event"),
                    budget_remaining_after=remaining,
                ))

        elif event.event_type == SessionEventType.TOOL_CALL:
            tools_invoked.append(ToolInvocation(
                tool_name=data.get("tool_name", "unknown"),
                timestamp=event.timestamp,
                arguments=data.get("arguments", {}),
                result_summary=data.get("result_summary"),
                provenance=event.provenance.value,
            ))

        elif event.event_type == SessionEventType.PERMISSION_REQUEST:
            permissions_requested.extend(data.get("permissions", []))

        elif event.event_type == SessionEventType.PERMISSION_GRANT:
            permissions_granted_during.extend(data.get("permissions", []))

        elif event.event_type == SessionEventType.EXTERNAL_CONTACT:
            pass  # Domains tracked at session level
