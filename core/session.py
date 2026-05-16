"""Workload session tracking for Barrikada agentic security.

Provides a session object that persists across the lifecycle of a single
agent task, recording declared intent, tool calls, permissions, external
contacts, input provenance, and pipeline events.

The store is defined as an abstract interface (``SessionStoreBackend``) so
that the in-memory implementation can be swapped for Redis/Postgres later
with zero changes to consuming code.
"""

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np

from core.session_settings import SessionSettings
from models.verdicts import InputProvenance

log = logging.getLogger(__name__)


#Event Types

class SessionEventType(str, Enum):
    """Types of events recorded in a workload session."""

    TOOL_CALL = "tool_call"
    PERMISSION_REQUEST = "permission_request"
    PERMISSION_GRANT = "permission_grant"
    EXTERNAL_CONTACT = "external_contact"
    PIPELINE_RESULT = "pipeline_result"
    INTERVENTION = "intervention"
    SUB_AGENT_SPAWN = "sub_agent_spawn"
    ESCALATION = "escalation"
    DRIFT_CHECK = "drift_check"
    RISK_BUDGET_DEDUCTION = "risk_budget_deduction"


class SessionStatus(str, Enum):
    """Lifecycle status of a workload session."""

    ACTIVE = "active"
    PAUSED = "paused"          # Budget exhausted, awaiting human review
    COMPLETED = "completed"    # Session closed normally
    HALTED = "halted"          # Stopped by an intervention


class SessionNotActiveError(RuntimeError):
    """Raised when a detect/mutate call is made on a session that is no longer active.

    The session's current status is exposed as ``status`` so callers (e.g. the
    HTTP layer) can map it to an appropriate response.
    """

    def __init__(self, session_id: str, status: "SessionStatus"):
        self.session_id = session_id
        self.status = status
        super().__init__(
            f"Session {session_id} is not active (status={status.value}); "
            "human review or a new session is required before proceeding."
        )


#Data Classes


@dataclass
class SessionEvent:
    """A single timestamped event within a workload session."""

    event_id: str
    event_type: SessionEventType
    timestamp: datetime
    data: dict[str, Any]
    provenance: InputProvenance = InputProvenance.UNKNOWN

    # Optional back-reference to a PipelineResult (stored as dict to
    # avoid circular imports and keep serialization simple).
    pipeline_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "provenance": self.provenance.value,
            "pipeline_result": self.pipeline_result,
        }


@dataclass
class WorkloadSession:
    """State for a single agent workload session.

    Attributes:
        delegation_chain: List of agent identifiers representing the
            delegation path.  Populated by the calling framework;
            Barrikada does not validate chain integrity.  Cryptographic
            verification of delegation chains is out of scope for this
            phase.
    """

    session_id: str
    declared_intent: str
    intent_vector: np.ndarray
    created_at: datetime
    status: SessionStatus = SessionStatus.ACTIVE

    # Delegation (populated by the calling framework, not validated)
    delegation_chain: list[str] = field(default_factory=list)

    # Event log
    events: list[SessionEvent] = field(default_factory=list)

    # Permissions
    permissions_granted: list[str] = field(default_factory=list)
    permissions_requested: list[str] = field(default_factory=list)

    # External contacts
    external_domains_contacted: list[str] = field(default_factory=list)

    # Risk budget
    risk_budget_initial: int = 5
    risk_budget_remaining: int = 5

    # Tracking
    closed_at: datetime | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        """Lightweight summary without the full event log."""
        return {
            "session_id": self.session_id,
            "declared_intent": self.declared_intent,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "event_count": len(self.events),
            "permissions_granted": list(self.permissions_granted),
            "permissions_requested": list(self.permissions_requested),
            "external_domains_contacted": list(self.external_domains_contacted),
            "delegation_chain": list(self.delegation_chain),
            "risk_budget_initial": self.risk_budget_initial,
            "risk_budget_remaining": self.risk_budget_remaining,
        }


#Abstract Backend 


class SessionStoreBackend(ABC):
    """Abstract interface for session storage.

    TODO: Implement this to swap the in-memory store for Redis, Postgres, etc.
    All consuming code depends on this interface — not a concrete store.
    """

    @abstractmethod
    def create_session(
        self,
        declared_intent: str,
        intent_vector: np.ndarray,
        initial_permissions: list[str] | None = None,
        delegation_chain: list[str] | None = None,
        risk_budget: int | None = None,
    ) -> WorkloadSession:
        ...

    @abstractmethod
    def get_session(self, session_id: str) -> WorkloadSession | None:
        ...

    @abstractmethod
    def append_event(self, session_id: str, event: SessionEvent) -> None:
        ...

    @abstractmethod
    def update_risk_budget(self, session_id: str, delta: int) -> int:
        """Deduct ``delta`` from the session's remaining budget.

        Returns the new remaining budget (may be negative).
        """
        ...

    @abstractmethod
    def set_session_status(self, session_id: str, status: SessionStatus) -> None:
        ...

    @abstractmethod
    def close_session(self, session_id: str) -> WorkloadSession:
        ...

    @abstractmethod
    def list_active_sessions(self) -> list[str]:
        ...


#In-Memory Implementation ───────────────────────────────────────────


class InMemorySessionStore(SessionStoreBackend):
    """Thread-safe in-memory session store.

    Sessions are evicted lazily when accessed after their TTL expires.

    .. note::

        Data is lost on process restart.  This is a known limitation of
        the in-memory backend and is acceptable for the MVP phase.

    .. todo::

        Replace with a Redis or Postgres-backed implementation for
        production deployments that require persistence across restarts,
        horizontal scaling, or shared state between API replicas.
    """

    def __init__(self, settings: SessionSettings | None = None) -> None:
        self._settings = settings or SessionSettings()
        self._sessions: dict[str, WorkloadSession] = {}
        self._lock = threading.Lock()

    #helpers ──

    def _is_expired(self, session: WorkloadSession) -> bool:
        age = (datetime.now(timezone.utc) - session.created_at).total_seconds()
        return age > self._settings.session_ttl_seconds

    def _evict_expired(self) -> None:
        """Remove sessions older than TTL.  Called under lock."""
        expired = [
            sid
            for sid, s in self._sessions.items()
            if self._is_expired(s)
        ]
        for sid in expired:
            log.info("Evicting expired session %s", sid)
            del self._sessions[sid]

    #public API

    def create_session(
        self,
        declared_intent: str,
        intent_vector: np.ndarray,
        initial_permissions: list[str] | None = None,
        delegation_chain: list[str] | None = None,
        risk_budget: int | None = None,
    ) -> WorkloadSession:
        budget = risk_budget if risk_budget is not None else self._settings.default_risk_budget
        session = WorkloadSession(
            session_id=uuid4().hex,
            declared_intent=declared_intent,
            intent_vector=intent_vector,
            created_at=datetime.now(timezone.utc),
            permissions_granted=list(initial_permissions or []),
            delegation_chain=list(delegation_chain or []),
            risk_budget_initial=budget,
            risk_budget_remaining=budget,
        )
        with self._lock:
            self._evict_expired()
            self._sessions[session.session_id] = session
            # TODO: Publish session-created event for external observability
        log.info("Created session %s (budget=%d)", session.session_id, budget)
        return session

    def get_session(self, session_id: str) -> WorkloadSession | None:
        with self._lock:
            self._evict_expired()
            # TODO: For Redis backend, deserialize from external store here
            return self._sessions.get(session_id)

    def append_event(self, session_id: str, event: SessionEvent) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")
            if len(session.events) >= self._settings.max_events_per_session:
                log.warning(
                    "Session %s hit max events (%d), dropping event %s",
                    session_id,
                    self._settings.max_events_per_session,
                    event.event_id,
                )
                return
            session.events.append(event)
            # TODO: For Redis backend, persist the event to external store here

    def update_risk_budget(self, session_id: str, delta: int) -> int:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")
            session.risk_budget_remaining -= delta
            # TODO: For Redis backend, persist budget change atomically here
            return session.risk_budget_remaining

    def set_session_status(self, session_id: str, status: SessionStatus) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")
            session.status = status
            # TODO: For Redis backend, persist status change here

    def close_session(self, session_id: str) -> WorkloadSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session {session_id} not found")
            session.status = SessionStatus.COMPLETED
            session.closed_at = datetime.now(timezone.utc)
            # TODO: For Redis backend, mark session as closed and optionally
            # archive to a durable store (Postgres) for long-term reporting
            return session

    def list_active_sessions(self) -> list[str]:
        with self._lock:
            self._evict_expired()
            return [
                sid
                for sid, s in self._sessions.items()
                if s.status == SessionStatus.ACTIVE
            ]
