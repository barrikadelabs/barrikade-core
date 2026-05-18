"""Risk Budget Engine — stateful per-session risk counter.

Each workload session gets a budget of "risky actions" it can take before
triggering mandatory human review.  The engine classifies each action by
risk category, deducts the configured cost, and returns an intervention
when the budget is exhausted.

Category costs are configurable via ``SessionSettings.category_costs`` and
the total budget can be overridden per-session at creation time.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from core.session import (
    SessionEvent,
    SessionEventType,
    SessionStoreBackend,
    SessionStatus,
)
from core.session_settings import SessionSettings
from models.verdicts import Intervention

log = logging.getLogger(__name__)


class RiskCategory(str, Enum):
    """Categories of security-exposure-increasing actions."""

    PERMISSION_EXPANSION = "permission_expansion"
    NEW_EXTERNAL_DOMAIN = "new_external_domain"
    UNTRUSTED_TO_TRUSTED_DATA_FLOW = "untrusted_to_trusted_data_flow"
    IRREVERSIBLE_ACTION = "irreversible_action"
    HIGH_INTENT_DRIFT = "high_intent_drift"
    PIPELINE_FLAG = "pipeline_flag"


@dataclass
class RiskEvent:
    """A single risk-budget-consuming event."""

    category: RiskCategory
    cost: int
    description: str
    timestamp: datetime


@dataclass
class RiskAssessment:
    """Result of a risk budget check.

    Attributes:
        allowed: Whether the action is permitted under the current budget.
        budget_remaining: Remaining budget after this assessment.
        budget_total: The session's initial total budget.
        intervention: The intervention to apply (NONE if allowed).
        risk_events: Individual risk events that were assessed.
        reason: Human-readable explanation.
    """

    allowed: bool
    budget_remaining: int
    budget_total: int
    intervention: Intervention
    risk_events: list[RiskEvent]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "budget_remaining": self.budget_remaining,
            "budget_total": self.budget_total,
            "intervention": self.intervention.value,
            "risk_events": [
                {
                    "category": e.category.value,
                    "cost": e.cost,
                    "description": e.description,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in self.risk_events
            ],
            "reason": self.reason,
        }


class RiskBudgetEngine:
    """Stateful per-session risk budget checker.

    Called by the session orchestrator before each action.  When the budget
    is exhausted the engine returns ``Intervention.ESCALATE`` by default.
    """

    def __init__(
        self,
        session_store: SessionStoreBackend,
        settings: SessionSettings | None = None,
    ) -> None:
    
        self._store = session_store
        self._settings = settings or SessionSettings()

    def _category_cost(self, category: RiskCategory) -> int:
        """Look up the configured cost for a risk category.""" 
        return self._settings.category_costs.get(category.value, 1)

    def assess_risk(
        self,
        session_id: str,
        categories: list[RiskCategory],
        descriptions: list[str] | None = None,
    ) -> RiskAssessment:
        """Assess risk for one or more categories and deduct from budget.

        Args:
            session_id: The active session to charge.
            categories: Risk categories triggered by this action.
            descriptions: Optional human-readable description per category.

        Returns:
            RiskAssessment with the intervention decision.
        """
        session = self._store.get_session(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} not found")

        descs = descriptions or [""] * len(categories)
        now = datetime.now(timezone.utc)

        risk_events: list[RiskEvent] = []
        total_cost = 0

        for cat, desc in zip(categories, descs):
            cost = self._category_cost(cat)
            if cost <= 0:
                continue
            risk_events.append(RiskEvent(
                category=cat,
                cost=cost,
                description=desc or f"{cat.value} detected",
                timestamp=now,
            ))
            total_cost += cost

        if total_cost == 0:
            # No risk cost — nothing to deduct
            return RiskAssessment(
                allowed=True,
                budget_remaining=session.risk_budget_remaining,
                budget_total=session.risk_budget_initial,
                intervention=Intervention.NONE,
                risk_events=risk_events,
                reason="No risk-consuming actions in this step.",
            )

        # Deduct from budget
        new_remaining = self._store.update_risk_budget(session_id, total_cost)

        # Record the deduction as a session event
        self._store.append_event(
            session_id,
            SessionEvent(
                event_id=f"risk_{now.timestamp():.0f}",
                event_type=SessionEventType.RISK_BUDGET_DEDUCTION,
                timestamp=now,
                data={
                    "categories": [e.category.value for e in risk_events],
                    "total_cost": total_cost,
                    "budget_remaining": new_remaining,
                },
            ),
        )

        if new_remaining <= 0:
            # Budget exhausted — escalate
            log.warning(
                "Session %s risk budget exhausted (remaining=%d), escalating",
                session_id,
                new_remaining,
            )
            self._store.set_session_status(session_id, SessionStatus.PAUSED)
            return RiskAssessment(
                allowed=False,
                budget_remaining=new_remaining,
                budget_total=session.risk_budget_initial,
                intervention=Intervention.ESCALATE,
                risk_events=risk_events,
                reason=(
                    f"Risk budget exhausted (spent {total_cost}, "
                    f"remaining {new_remaining}/{session.risk_budget_initial}). "
                    "Mandatory human review required before proceeding."
                ),
            )

        return RiskAssessment(
            allowed=True,
            budget_remaining=new_remaining,
            budget_total=session.risk_budget_initial,
            intervention=Intervention.NONE,
            risk_events=risk_events,
            reason=(
                f"Risk budget OK ({new_remaining}/{session.risk_budget_initial} "
                f"remaining after deducting {total_cost})."
            ),
        )
