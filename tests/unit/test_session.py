"""Unit tests for the Workload Session Tracker."""

import time
from datetime import datetime, timezone

import numpy as np
import pytest

from core.session import (
    InMemorySessionStore,
    SessionEvent,
    SessionEventType,
    SessionStatus,
    WorkloadSession,    
)
from core.session_settings import SessionSettings
from models.verdicts import InputProvenance


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def settings():
    return SessionSettings(session_ttl_seconds=5)


@pytest.fixture
def store(settings):
    return InMemorySessionStore(settings)


def _dummy_vector() -> np.ndarray:
    """Create a unit-normalised dummy intent vector."""
    vec = np.random.randn(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


# ── Session Lifecycle ───────────────────────────────────────────────────


def test_create_session(store):
    session = store.create_session(
        declared_intent="Summarise quarterly report",
        intent_vector=_dummy_vector(),
        initial_permissions=["read_files"],
        delegation_chain=["user_1", "agent_alpha"],
        risk_budget=10,
    )
    assert isinstance(session, WorkloadSession)
    assert session.status == SessionStatus.ACTIVE
    assert session.risk_budget_initial == 10
    assert session.risk_budget_remaining == 10
    assert session.permissions_granted == ["read_files"]
    assert session.delegation_chain == ["user_1", "agent_alpha"]
    assert session.declared_intent == "Summarise quarterly report"


def test_get_session(store):
    session = store.create_session(
        declared_intent="test",
        intent_vector=_dummy_vector(),
    )
    retrieved = store.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.session_id == session.session_id


def test_get_nonexistent_session(store):
    assert store.get_session("nonexistent") is None


def test_close_session(store):
    session = store.create_session(
        declared_intent="test",
        intent_vector=_dummy_vector(),
    )
    closed = store.close_session(session.session_id)
    assert closed.status == SessionStatus.COMPLETED
    assert closed.closed_at is not None


def test_close_nonexistent_session(store):
    with pytest.raises(KeyError):
        store.close_session("nonexistent")


def test_list_active_sessions(store):
    s1 = store.create_session("task 1", _dummy_vector())
    s2 = store.create_session("task 2", _dummy_vector())
    s3 = store.create_session("task 3", _dummy_vector())

    active = store.list_active_sessions()
    assert len(active) == 3

    store.close_session(s2.session_id)
    active = store.list_active_sessions()
    assert len(active) == 2
    assert s2.session_id not in active


def test_default_risk_budget():
    settings = SessionSettings(default_risk_budget=7)
    store = InMemorySessionStore(settings)
    session = store.create_session("test", _dummy_vector())
    assert session.risk_budget_initial == 7
    assert session.risk_budget_remaining == 7


def test_risk_budget_override(store):
    session = store.create_session(
        declared_intent="test",
        intent_vector=_dummy_vector(),
        risk_budget=20,
    )
    assert session.risk_budget_initial == 20


# ── Event Appending ─────────────────────────────────────────────────────


def test_append_event(store):
    session = store.create_session("test", _dummy_vector())
    event = SessionEvent(
        event_id="evt_001",
        event_type=SessionEventType.TOOL_CALL,
        timestamp=datetime.now(timezone.utc),
        data={"tool_name": "read_file"},
        provenance=InputProvenance.TRUSTED_INTERNAL,
    )
    store.append_event(session.session_id, event)

    retrieved = store.get_session(session.session_id)
    assert retrieved is not None
    assert len(retrieved.events) == 1
    assert retrieved.events[0].event_id == "evt_001"


def test_append_event_nonexistent_session(store):
    event = SessionEvent(
        event_id="evt_001",
        event_type=SessionEventType.TOOL_CALL,
        timestamp=datetime.now(timezone.utc),
        data={},
    )
    with pytest.raises(KeyError):
        store.append_event("nonexistent", event)


def test_max_events_cap():
    settings = SessionSettings(max_events_per_session=3)
    store = InMemorySessionStore(settings)
    session = store.create_session("test", _dummy_vector())

    for i in range(5):
        event = SessionEvent(
            event_id=f"evt_{i}",
            event_type=SessionEventType.TOOL_CALL,
            timestamp=datetime.now(timezone.utc),
            data={},
        )
        store.append_event(session.session_id, event)

    retrieved = store.get_session(session.session_id)
    assert retrieved is not None
    assert len(retrieved.events) == 3  # capped


# ── Risk Budget Updates ─────────────────────────────────────────────────


def test_update_risk_budget(store):
    session = store.create_session(
        "test", _dummy_vector(), risk_budget=5
    )
    remaining = store.update_risk_budget(session.session_id, 2)
    assert remaining == 3

    remaining = store.update_risk_budget(session.session_id, 4)
    assert remaining == -1  # Can go negative


def test_update_budget_nonexistent(store):
    with pytest.raises(KeyError):
        store.update_risk_budget("nonexistent", 1)


# ── Session Status ──────────────────────────────────────────────────────


def test_set_session_status(store):
    session = store.create_session("test", _dummy_vector())
    store.set_session_status(session.session_id, SessionStatus.PAUSED)

    retrieved = store.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.status == SessionStatus.PAUSED


# ── TTL Eviction ────────────────────────────────────────────────────────


def test_ttl_eviction():
    settings = SessionSettings(session_ttl_seconds=1)
    store = InMemorySessionStore(settings)
    session = store.create_session("test", _dummy_vector())

    assert store.get_session(session.session_id) is not None

    time.sleep(1.5)

    assert store.get_session(session.session_id) is None


# ── Serialization ───────────────────────────────────────────────────────


def test_session_summary_dict(store):
    session = store.create_session(
        declared_intent="test task",
        intent_vector=_dummy_vector(),
        initial_permissions=["read"],
    )
    summary = session.to_summary_dict()
    assert summary["session_id"] == session.session_id
    assert summary["declared_intent"] == "test task"
    assert summary["status"] == "active"
    assert summary["permissions_granted"] == ["read"]


def test_event_to_dict():
    event = SessionEvent(
        event_id="evt_001",
        event_type=SessionEventType.TOOL_CALL,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        data={"tool_name": "read_file"},
        provenance=InputProvenance.TRUSTED_INTERNAL,
    )
    d = event.to_dict()
    assert d["event_type"] == "tool_call"
    assert d["provenance"] == "trusted_internal"


# ── Thread Safety ───────────────────────────────────────────────────────


def test_concurrent_event_appending(store):
    """Verify thread safety by appending events from multiple threads."""
    import threading

    session = store.create_session("concurrent test", _dummy_vector())
    errors: list[Exception] = []

    def append_events(start_idx):
        try:
            for i in range(50):
                event = SessionEvent(
                    event_id=f"evt_{start_idx}_{i}",
                    event_type=SessionEventType.TOOL_CALL,
                    timestamp=datetime.now(timezone.utc),
                    data={},
                )
                store.append_event(session.session_id, event)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=append_events, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    retrieved = store.get_session(session.session_id)
    assert retrieved is not None
    assert len(retrieved.events) == 250  # 5 threads × 50 events
