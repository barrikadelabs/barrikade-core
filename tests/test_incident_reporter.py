"""Unit tests for the Incident Reporter."""

from datetime import datetime, timezone

import numpy as np
import pytest

from core.incident_reporter import IncidentReporter
from core.session import (
    InMemorySessionStore,
    SessionEvent,
    SessionEventType,
)
from core.session_settings import SessionSettings
from models.incident_report import IncidentReport
from models.verdicts import InputProvenance


# ── Fixtures ────────────────────────────────────────────────────────────


def _dummy_vector() -> np.ndarray:
    vec = np.random.randn(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def settings():
    return SessionSettings()


@pytest.fixture
def store(settings):
    return InMemorySessionStore(settings)


@pytest.fixture
def reporter(store):
    return IncidentReporter(store)


def _make_session(store, intent="test task"):
    return store.create_session(
        declared_intent=intent,
        intent_vector=_dummy_vector(),
        initial_permissions=["read_files", "write_logs"],
        delegation_chain=["user_1", "agent_alpha"],
        risk_budget=5,
    )


# ── Basic Report Generation ────────────────────────────────────────────


def test_generate_report_empty_session(store, reporter):
    session = _make_session(store)
    store.close_session(session.session_id)

    report = reporter.generate_report(session.session_id)
    assert isinstance(report, IncidentReport)
    assert report.workload_id == session.session_id
    assert report.declared_task == "test task"
    assert report.delegation_chain == ["user_1", "agent_alpha"]
    assert report.permissions_initially_granted == ["read_files", "write_logs"]
    assert report.is_near_miss is False
    assert report.final_outcome == "completed"
    assert report.risk_budget_initial == 5
    assert report.barrikade_version == "0.1.0"


def test_generate_report_nonexistent_session(reporter):
    with pytest.raises(KeyError):
        reporter.generate_report("nonexistent")


# ── Pipeline Events ────────────────────────────────────────────────────


def test_pipeline_events_recorded(store, reporter):
    session = _make_session(store)
    now = datetime.now(timezone.utc)

    store.append_event(session.session_id, SessionEvent(
        event_id="pe_001",
        event_type=SessionEventType.PIPELINE_RESULT,
        timestamp=now,
        data={"source": "test"},
        provenance=InputProvenance.UNTRUSTED_EXTERNAL,
        pipeline_result={
            "input_hash": "abc123",
            "final_verdict": "allow",
            "decision_layer": "B",
            "confidence_score": 0.95,
            "total_processing_time_ms": 12.5,
        },
    ))

    store.close_session(session.session_id)
    report = reporter.generate_report(session.session_id)

    assert len(report.pipeline_events) == 1
    assert report.pipeline_events[0].final_verdict == "allow"
    assert report.pipeline_events[0].input_hash == "abc123"
    assert len(report.inputs) == 1
    assert report.inputs[0].trust_level == "untrusted_external"


# ── Near-Miss Detection ────────────────────────────────────────────────


def test_near_miss_on_pipeline_block(store, reporter):
    session = _make_session(store)
    now = datetime.now(timezone.utc)

    store.append_event(session.session_id, SessionEvent(
        event_id="pe_block",
        event_type=SessionEventType.PIPELINE_RESULT,
        timestamp=now,
        data={"source": "test"},
        pipeline_result={
            "input_hash": "mal_001",
            "final_verdict": "block",
            "decision_layer": "B",
            "confidence_score": 0.98,
            "total_processing_time_ms": 8.0,
        },
    ))

    store.close_session(session.session_id)
    report = reporter.generate_report(session.session_id)

    assert report.is_near_miss is True
    assert "block" in report.near_miss_details


def test_near_miss_on_pipeline_flag(store, reporter):
    session = _make_session(store)
    now = datetime.now(timezone.utc)

    store.append_event(session.session_id, SessionEvent(
        event_id="pe_flag",
        event_type=SessionEventType.PIPELINE_RESULT,
        timestamp=now,
        data={"source": "test"},
        pipeline_result={
            "input_hash": "sus_001",
            "final_verdict": "flag",
            "decision_layer": "C",
            "confidence_score": 0.70,
            "total_processing_time_ms": 15.0,
        },
    ))

    store.close_session(session.session_id)
    report = reporter.generate_report(session.session_id)
    assert report.is_near_miss is True


def test_near_miss_on_escalate_intervention(store, reporter):
    session = _make_session(store)
    now = datetime.now(timezone.utc)

    store.append_event(session.session_id, SessionEvent(
        event_id="iv_001",
        event_type=SessionEventType.INTERVENTION,
        timestamp=now,
        data={
            "intervention": "escalate",
            "reason": "Budget exhausted",
            "trigger": "risk_budget",
        },
    ))

    store.close_session(session.session_id)
    report = reporter.generate_report(session.session_id)
    assert report.is_near_miss is True
    assert len(report.interventions) == 1


# ── Drift Events ───────────────────────────────────────────────────────


def test_drift_events_recorded(store, reporter):
    session = _make_session(store)
    now = datetime.now(timezone.utc)

    store.append_event(session.session_id, SessionEvent(
        event_id="drift_001",
        event_type=SessionEventType.DRIFT_CHECK,
        timestamp=now,
        data={
            "drift_score": 0.42,
            "cosine_similarity": 0.58,
            "risk_level": "high",
            "proposed_action_summary": "connect to external API",
        },
    ))

    store.close_session(session.session_id)
    report = reporter.generate_report(session.session_id)

    assert len(report.drift_events) == 1
    assert report.max_intent_drift_score == 0.42
    assert report.drift_events[0].risk_level == "high"


# ── Tool Calls ──────────────────────────────────────────────────────────


def test_tool_calls_recorded(store, reporter):
    session = _make_session(store)
    now = datetime.now(timezone.utc)

    store.append_event(session.session_id, SessionEvent(
        event_id="tool_001",
        event_type=SessionEventType.TOOL_CALL,
        timestamp=now,
        data={
            "tool_name": "read_database",
            "arguments": {"query": "SELECT *"},
            "result_summary": "100 rows",
        },
        provenance=InputProvenance.TRUSTED_INTERNAL,
    ))

    store.close_session(session.session_id)
    report = reporter.generate_report(session.session_id)

    assert len(report.tools_invoked) == 1
    assert report.tools_invoked[0].tool_name == "read_database"
    assert report.tools_invoked[0].provenance == "trusted_internal"


# ── Risk Budget Events ─────────────────────────────────────────────────


def test_risk_events_recorded(store, reporter):
    session = _make_session(store)
    now = datetime.now(timezone.utc)

    store.append_event(session.session_id, SessionEvent(
        event_id="risk_001",
        event_type=SessionEventType.RISK_BUDGET_DEDUCTION,
        timestamp=now,
        data={
            "categories": ["new_external_domain"],
            "total_cost": 1,
            "budget_remaining": 4,
        },
    ))

    store.close_session(session.session_id)
    report = reporter.generate_report(session.session_id)

    assert len(report.risk_events) == 1
    assert report.risk_events[0].category == "new_external_domain"


# ── Permissions ─────────────────────────────────────────────────────────


def test_permission_events_recorded(store, reporter):
    session = _make_session(store)
    now = datetime.now(timezone.utc)

    store.append_event(session.session_id, SessionEvent(
        event_id="perm_req_001",
        event_type=SessionEventType.PERMISSION_REQUEST,
        timestamp=now,
        data={"permissions": ["write_database"]},
    ))
    store.append_event(session.session_id, SessionEvent(
        event_id="perm_grant_001",
        event_type=SessionEventType.PERMISSION_GRANT,
        timestamp=now,
        data={"permissions": ["write_database"]},
    ))

    store.close_session(session.session_id)
    report = reporter.generate_report(session.session_id)

    assert "write_database" in report.permissions_later_requested
    assert "write_database" in report.permissions_granted_during_session


# ── Serialization ───────────────────────────────────────────────────────


def test_export_json(store, reporter):
    session = _make_session(store)
    store.close_session(session.session_id)
    report = reporter.generate_report(session.session_id)

    json_str = reporter.export_json(report)
    assert isinstance(json_str, str)
    assert session.session_id in json_str

    import json
    parsed = json.loads(json_str)
    assert parsed["workload_id"] == session.session_id


def test_export_dict(store, reporter):
    session = _make_session(store)
    store.close_session(session.session_id)
    report = reporter.generate_report(session.session_id)

    d = reporter.export_dict(report)
    assert isinstance(d, dict)
    assert d["workload_id"] == session.session_id
    assert d["is_near_miss"] is False


# ── Duration Tracking ──────────────────────────────────────────────────


def test_session_duration(store, reporter):
    session = _make_session(store)
    store.close_session(session.session_id)
    report = reporter.generate_report(session.session_id)

    assert report.session_started_at is not None
    assert report.session_ended_at is not None
    assert report.total_duration_seconds is not None
    assert report.total_duration_seconds >= 0
