"""Unit tests for the Risk Budget Engine."""

import numpy as np
import pytest

from core.risk_budget import RiskAssessment, RiskBudgetEngine, RiskCategory
from core.session import InMemorySessionStore, SessionStatus
from core.session_settings import SessionSettings
from models.verdicts import Intervention


# ── Fixtures ────────────────────────────────────────────────────────────


def _dummy_vector() -> np.ndarray:
    vec = np.random.randn(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def settings():
    return SessionSettings(
        default_risk_budget=5,
        category_costs={
            "permission_expansion": 1,
            "new_external_domain": 1,
            "untrusted_to_trusted_data_flow": 1,
            "irreversible_action": 2,
            "high_intent_drift": 1,
            "pipeline_flag": 1,
        },
    )


@pytest.fixture
def store(settings):
    return InMemorySessionStore(settings)


@pytest.fixture
def engine(store, settings):
    return RiskBudgetEngine(store, settings)


@pytest.fixture
def session_id(store):
    session = store.create_session(
        declared_intent="test task",
        intent_vector=_dummy_vector(),
        risk_budget=5,
    )
    return session.session_id


# ── Basic Assessment ────────────────────────────────────────────────────


def test_assess_single_risk(engine, session_id):
    result = engine.assess_risk(
        session_id,
        [RiskCategory.NEW_EXTERNAL_DOMAIN],
        ["Contact with api.example.com"],
    )
    assert isinstance(result, RiskAssessment)
    assert result.allowed is True
    assert result.budget_remaining == 4
    assert result.intervention == Intervention.NONE
    assert len(result.risk_events) == 1


def test_assess_no_risk(engine, session_id):
    result = engine.assess_risk(session_id, [], [])
    assert result.allowed is True
    assert result.budget_remaining == 5  # No deduction
    assert result.intervention == Intervention.NONE


def test_assess_multiple_categories(engine, session_id):
    result = engine.assess_risk(
        session_id,
        [RiskCategory.NEW_EXTERNAL_DOMAIN, RiskCategory.PIPELINE_FLAG],
        ["domain contact", "pipeline flagged"],
    )
    assert result.allowed is True
    assert result.budget_remaining == 3  # 5 - 1 - 1


# ── Budget Exhaustion ───────────────────────────────────────────────────


def test_budget_exhaustion_triggers_escalate(engine, session_id):
    # Deduct 4 (should be OK)
    for _ in range(4):
        result = engine.assess_risk(
            session_id,
            [RiskCategory.PIPELINE_FLAG],
            ["flagged"],
        )
        assert result.allowed is True

    # Deduct 1 more (budget = 0 → escalate)
    result = engine.assess_risk(
        session_id,
        [RiskCategory.PIPELINE_FLAG],
        ["final flag"],
    )
    assert result.allowed is False
    assert result.budget_remaining == 0
    assert result.intervention == Intervention.ESCALATE


def test_budget_goes_negative(engine, session_id):
    # Irreversible action costs 2
    # 5 - 2 = 3, 3 - 2 = 1, 1 - 2 = -1 → escalate
    engine.assess_risk(session_id, [RiskCategory.IRREVERSIBLE_ACTION], ["action 1"])
    engine.assess_risk(session_id, [RiskCategory.IRREVERSIBLE_ACTION], ["action 2"])
    result = engine.assess_risk(session_id, [RiskCategory.IRREVERSIBLE_ACTION], ["action 3"])

    assert result.allowed is False
    assert result.budget_remaining == -1
    assert result.intervention == Intervention.ESCALATE


def test_session_paused_on_budget_exhaustion(engine, session_id, store):
    # Exhaust budget
    for _ in range(5):
        engine.assess_risk(session_id, [RiskCategory.PIPELINE_FLAG], ["flag"])

    session = store.get_session(session_id)
    assert session is not None
    assert session.status == SessionStatus.PAUSED


# ── Category Costs ──────────────────────────────────────────────────────


def test_irreversible_action_costs_two(engine, session_id):
    result = engine.assess_risk(
        session_id,
        [RiskCategory.IRREVERSIBLE_ACTION],
        ["delete database"],
    )
    assert result.budget_remaining == 3  # 5 - 2


def test_zero_cost_category():
    """A category with cost 0 should not deduct from budget."""
    settings = SessionSettings(
        default_risk_budget=5,
        category_costs={"new_external_domain": 0},
    )
    store = InMemorySessionStore(settings)
    engine = RiskBudgetEngine(store, settings)
    session = store.create_session("test", _dummy_vector(), risk_budget=5)

    result = engine.assess_risk(
        session.session_id,
        [RiskCategory.NEW_EXTERNAL_DOMAIN],
        ["contact domain"],
    )
    assert result.allowed is True
    assert result.budget_remaining == 5  # No deduction


# ── Per-session Budget Override ─────────────────────────────────────────


def test_per_session_budget_override(store, settings):
    engine = RiskBudgetEngine(store, settings)
    session = store.create_session(
        "test", _dummy_vector(), risk_budget=2
    )

    engine.assess_risk(session.session_id, [RiskCategory.PIPELINE_FLAG], ["f1"])
    result = engine.assess_risk(
        session.session_id, [RiskCategory.PIPELINE_FLAG], ["f2"]
    )

    assert result.allowed is False  # 2 - 1 - 1 = 0 → escalate
    assert result.budget_remaining == 0


# ── Error Cases ─────────────────────────────────────────────────────────


def test_assess_nonexistent_session(engine):
    with pytest.raises(KeyError):
        engine.assess_risk("nonexistent", [RiskCategory.PIPELINE_FLAG], ["flag"])


# ── Serialization ───────────────────────────────────────────────────────


def test_risk_assessment_to_dict(engine, session_id):
    result = engine.assess_risk(
        session_id,
        [RiskCategory.NEW_EXTERNAL_DOMAIN],
        ["test domain"],
    )
    d = result.to_dict()
    assert "allowed" in d
    assert "budget_remaining" in d
    assert "intervention" in d
    assert d["intervention"] == "none"
    assert len(d["risk_events"]) == 1


# ── Event Recording ────────────────────────────────────────────────────


def test_risk_deduction_recorded_as_event(engine, session_id, store):
    engine.assess_risk(
        session_id,
        [RiskCategory.PIPELINE_FLAG],
        ["flagged"],
    )
    session = store.get_session(session_id)
    assert session is not None
    budget_events = [
        e for e in session.events
        if e.event_type.value == "risk_budget_deduction"
    ]
    assert len(budget_events) == 1
    assert budget_events[0].data["total_cost"] == 1
