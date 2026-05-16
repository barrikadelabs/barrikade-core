"""Session-aware orchestrator for Barrikada agentic security.

Wraps the existing stateless ``PIPipeline`` with session context, intent
drift detection, risk budget checking, and incident reporting.  The
original ``PIPipeline.detect()`` method is left completely unchanged —
this is a new orchestration layer on top.
"""

from core.session import InMemorySessionStore
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from core.incident_reporter import IncidentReporter
from core.intent_scorer import DriftResult, IntentDeviationScorer
from core.orchestrator import PIPipeline
from core.risk_budget import RiskAssessment, RiskBudgetEngine, RiskCategory
from core.session import (
    SessionEvent,
    SessionEventType,
    SessionNotActiveError,
    SessionStatus,
    SessionStoreBackend,
)
from core.session_settings import SessionSettings
from models.verdicts import InputProvenance, Intervention
from models.incident_report import IncidentReport

log = logging.getLogger(__name__)


@dataclass
class SessionDetectResult:
    """Composite result from a session-aware detection request.

    Wraps the raw ``PipelineResult`` (as dict) with session-level context:
    drift assessment, risk budget status, and any intervention applied.
    """

    # Original pipeline output (dict for serialization simplicity)
    pipeline_result: dict[str, Any]

    # Session context
    session_id: str
    drift: DriftResult | None = None
    risk_assessment: RiskAssessment | None = None
    intervention: Intervention = Intervention.NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_result": self.pipeline_result,
            "session_id": self.session_id,
            "drift": self.drift.to_dict() if self.drift else None,
            "risk_assessment": (
                self.risk_assessment.to_dict() if self.risk_assessment else None
            ),
            "intervention": self.intervention.value,
        }


class SessionOrchestrator:
    """Session-aware detection orchestrator.

    Composes the stateless pipeline with the four agentic security modules.
    Use the ``create_session_orchestrator()`` factory for convenient setup.
    """

    def __init__(
        self,
        pipeline: PIPipeline,
        session_store: SessionStoreBackend,
        intent_scorer: IntentDeviationScorer,
        risk_budget_engine: RiskBudgetEngine,
        incident_reporter: IncidentReporter,
        settings: SessionSettings | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._store = session_store
        self._scorer = intent_scorer
        self._budget = risk_budget_engine
        self._reporter = incident_reporter
        self._settings = settings or SessionSettings()

    # ── Session Lifecycle ───────────────────────────────────────────

    def start_session(
        self,
        declared_intent: str,
        permissions: list[str] | None = None,
        provenance: InputProvenance = InputProvenance.UNKNOWN,
        delegation_chain: list[str] | None = None,
        risk_budget: int | None = None,
    ) -> str:
        """Create a new workload session.

        Args:
            declared_intent: Free-text description of the agent's task.
            permissions: Initially granted permissions.
            provenance: Trust level of the session initiator.
            delegation_chain: List of agent identifiers in the delegation
                path (populated by the calling framework; Barrikada does
                not validate chain integrity).
            risk_budget: Override the default risk budget for this session.

        Returns:
            session_id for subsequent calls.
        """
        intent_vector = self._scorer.embed_intent(declared_intent)
        session = self._store.create_session(
            declared_intent=declared_intent,
            intent_vector=intent_vector,
            initial_permissions=permissions,
            delegation_chain=delegation_chain,
            risk_budget=risk_budget,
        )
        log.info(
            "Started session %s: intent=%r, budget=%d",
            session.session_id,
            declared_intent[:80],
            session.risk_budget_initial,
        )
        return session.session_id

    def detect_with_session(
        self,
        session_id: str,
        input_text: str,
        provenance: InputProvenance = InputProvenance.UNKNOWN,
        tool_name: str | None = None,
        target_domain: str | None = None,
    ) -> SessionDetectResult:
        """Run a session-aware detection.

        1. Run ``pipeline.detect(input_text)`` → PipelineResult
        2. Record as SessionEvent
        3. Compute intent drift
        4. Classify risk categories
        5. Run risk budget assessment
        6. Return composite result

        Args:
            session_id: Active session to attribute this detection to.
            input_text: The text to screen.
            provenance: Trust level of this input.
            tool_name: Name of the tool being invoked (if applicable).
            target_domain: External domain being contacted (if applicable).

        Returns:
            SessionDetectResult with pipeline output + session context.
        """
        session = self._store.get_session(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} not found")

        # Reject detects on non-active sessions. PAUSED means a prior call
        # exhausted the risk budget and the doc-stated policy is to require
        # human review before proceeding; COMPLETED/HALTED sessions are
        # closed for any further activity. Without this gate the orchestrator
        # would keep running detection on a paused session and the risk
        # budget would drift further negative on each call.
        if session.status != SessionStatus.ACTIVE:
            raise SessionNotActiveError(session_id, session.status)

        now = datetime.now(timezone.utc)

        # 1. Run the stateless pipeline
        pipeline_result = self._pipeline.detect(input_text)
        result_dict = pipeline_result.to_dict()

        # 2. Record as pipeline event
        self._store.append_event(
            session_id,
            SessionEvent(
                event_id=uuid4().hex[:12],
                event_type=SessionEventType.PIPELINE_RESULT,
                timestamp=now,
                data={"source": "detect_with_session"},
                provenance=provenance,
                pipeline_result=result_dict,
            ),
        )

        # Record tool call if applicable
        if tool_name:
            self._store.append_event(
                session_id,
                SessionEvent(
                    event_id=uuid4().hex[:12],
                    event_type=SessionEventType.TOOL_CALL,
                    timestamp=now,
                    data={"tool_name": tool_name},
                    provenance=provenance,
                ),
            )

        # Record external domain contact
        if target_domain:
            if target_domain not in session.external_domains_contacted:
                session.external_domains_contacted.append(target_domain)
            self._store.append_event(
                session_id,
                SessionEvent(
                    event_id=uuid4().hex[:12],
                    event_type=SessionEventType.EXTERNAL_CONTACT,
                    timestamp=now,
                    data={"domain": target_domain},
                    provenance=provenance,
                ),
            )

        # 3. Compute intent drift
        drift = self._scorer.compute_drift(session.intent_vector, input_text)
        self._store.append_event(
            session_id,
            SessionEvent(
                event_id=uuid4().hex[:12],
                event_type=SessionEventType.DRIFT_CHECK,
                timestamp=now,
                data=drift.to_dict() | {
                    "proposed_action_summary": input_text[:200],
                },
                provenance=provenance,
            ),
        )

        # 4. Classify risk categories
        risk_categories: list[RiskCategory] = []
        risk_descriptions: list[str] = []

        # Pipeline flagged or blocked
        if pipeline_result.final_verdict.value in ("flag", "block"):
            risk_categories.append(RiskCategory.PIPELINE_FLAG)
            risk_descriptions.append(
                f"Pipeline verdict: {pipeline_result.final_verdict.value} "
                f"(layer {pipeline_result.decision_layer.value})"
            )

        # High intent drift
        if drift.drift_score >= self._settings.intent_drift_warn_threshold:
            risk_categories.append(RiskCategory.HIGH_INTENT_DRIFT)
            risk_descriptions.append(
                f"Intent drift {drift.drift_score:.3f} exceeds threshold "
                f"{self._settings.intent_drift_warn_threshold}"
            )

        # New external domain
        if target_domain:
            risk_categories.append(RiskCategory.NEW_EXTERNAL_DOMAIN)
            risk_descriptions.append(f"Contact with external domain: {target_domain}")

        # Untrusted data flow
        if provenance == InputProvenance.UNTRUSTED_EXTERNAL and tool_name:
            risk_categories.append(RiskCategory.UNTRUSTED_TO_TRUSTED_DATA_FLOW)
            risk_descriptions.append(
                f"Untrusted external input flowing into tool: {tool_name}"
            )

        # 5. Risk budget assessment
        risk_assessment: RiskAssessment | None = None
        intervention: Intervention = Intervention.NONE

        if risk_categories:
            risk_assessment = self._budget.assess_risk(
                session_id, risk_categories, risk_descriptions
            )
            intervention = risk_assessment.intervention

            # Record intervention event if non-trivial
            if intervention != Intervention.NONE:
                self._store.append_event(
                    session_id,
                    SessionEvent(
                        event_id=uuid4().hex[:12],
                        event_type=SessionEventType.INTERVENTION,
                        timestamp=now,
                        data={
                            "intervention": intervention.value,
                            "reason": risk_assessment.reason,
                            "trigger": "risk_budget",
                            "details": {
                                "categories": [c.value for c in risk_categories],
                                "budget_remaining": risk_assessment.budget_remaining,
                            },
                        },
                        provenance=provenance,
                    ),
                )

        return SessionDetectResult(
            pipeline_result=result_dict,
            session_id=session_id,
            drift=drift,
            risk_assessment=risk_assessment,
            intervention=intervention,
        )

    def end_session(
        self,
        session_id: str,
        model_version: str = "",
        scaffold_version: str = "",
    ) -> IncidentReport:
        """Close a session and generate the incident report.

        Args:
            session_id: The session to close.
            model_version: Model version for the report metadata.
            scaffold_version: Scaffold version for the report metadata.

        Returns:
            IncidentReport for the closed session.
        """
        self._store.close_session(session_id)
        report = self._reporter.generate_report(
            session_id,
            model_version=model_version,
            scaffold_version=scaffold_version,
        )
        log.info(
            "Session %s closed — near_miss=%s, drift_max=%.3f, budget=%d/%d",
            session_id,
            report.is_near_miss,
            report.max_intent_drift_score,
            report.risk_budget_final,
            report.risk_budget_initial,
        )
        return report

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """Get a lightweight summary of a session's current state."""
        session = self._store.get_session(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} not found")
        return session.to_summary_dict()


# Factory


def create_session_orchestrator(
    settings: SessionSettings | None = None,
    pipeline: PIPipeline | None = None,
) -> SessionOrchestrator:
    """Factory function for ``SessionOrchestrator``.

    Wires up all five composed dependencies in one place.  Keeps tests
    clean and makes the wiring explicit.

    Args:
        settings: Session configuration (uses defaults if None).
        pipeline: Existing PIPipeline instance to reuse.  If None, a new
            one is created (which triggers model loading).

    Returns:
        Fully configured SessionOrchestrator.
    """
    session_settings = settings or SessionSettings()

    # TODO: Replace InMemorySessionStore with Redis/Postgres backend for
    # production deployments requiring persistence or horizontal scaling.
    store = InMemorySessionStore(session_settings)

    resolved_pipeline = pipeline or PIPipeline()
    scorer = IntentDeviationScorer(settings=session_settings)
    budget_engine = RiskBudgetEngine(store, session_settings)
    reporter = IncidentReporter(store)

    return SessionOrchestrator(
        pipeline=resolved_pipeline,
        session_store=store,
        intent_scorer=scorer,
        risk_budget_engine=budget_engine,
        incident_reporter=reporter,
        settings=session_settings,
    )
