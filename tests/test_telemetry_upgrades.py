import json
import logging
import threading
import time
from unittest.mock import MagicMock
import pytest
from core.settings import Settings
from core.telemetry import TelemetryEngine
from core.session import InMemorySessionStore
from core.session_orchestrator import SessionOrchestrator
from models.verdicts import InputProvenance, FinalVerdict, DecisionLayer
from models.PipelineResult import PipelineResult


# ==========================================
# Upgrade 1: Sampling Tests
# ==========================================

def test_telemetry_sampling_anomalous_always_logged(tmp_path):
    custom_settings = Settings()
    log_file = tmp_path / "telemetry_sampling.jsonl"
    custom_settings.telemetry_log_path = str(log_file)
    custom_settings.telemetry_enabled = True
    custom_settings.telemetry_safe_sample_rate = 0.0  # 0% safe sample rate

    engine = TelemetryEngine(settings=custom_settings)

    # 1. Anomalous event should be logged even with 0% safe rate
    engine.emit_sampled(
        is_anomalous=True,
        event_type="pipeline_run",
        trace_id="4a3b2c",
        payload={"verdict": "block"},
    )
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["payload"]["verdict"] == "block"


def test_telemetry_sampling_safe_dropped_when_zero_rate(tmp_path):
    custom_settings = Settings()
    log_file = tmp_path / "telemetry_sampling_dropped.jsonl"
    custom_settings.telemetry_log_path = str(log_file)
    custom_settings.telemetry_enabled = True
    custom_settings.telemetry_safe_sample_rate = 0.0  # 0% safe sample rate

    engine = TelemetryEngine(settings=custom_settings)

    # 2. Safe event should be dropped
    engine.emit_sampled(
        is_anomalous=False,
        event_type="pipeline_run",
        trace_id="4a3b2c",
        payload={"verdict": "allow"},
    )
    assert not log_file.exists()


def test_telemetry_sampling_safe_logged_when_full_rate(tmp_path):
    custom_settings = Settings()
    log_file = tmp_path / "telemetry_sampling_full.jsonl"
    custom_settings.telemetry_log_path = str(log_file)
    custom_settings.telemetry_enabled = True
    custom_settings.telemetry_safe_sample_rate = 1.0  # 100% safe sample rate

    engine = TelemetryEngine(settings=custom_settings)

    # 3. Safe event should be logged
    engine.emit_sampled(
        is_anomalous=False,
        event_type="pipeline_run",
        trace_id="4a3b2c",
        payload={"verdict": "allow"},
    )
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["payload"]["verdict"] == "allow"
    # Verify metric sampled is injected
    assert parsed["metrics"]["sampled"] is True


def test_telemetry_sampling_deterministic_trace_id(tmp_path):
    custom_settings = Settings()
    log_file = tmp_path / "telemetry_sampling_deterministic.jsonl"
    custom_settings.telemetry_log_path = str(log_file)
    custom_settings.telemetry_enabled = True
    custom_settings.telemetry_safe_sample_rate = 0.5  # 50% sample rate

    engine = TelemetryEngine(settings=custom_settings)

    # Deterministic sampling check based on trace_id hash
    trace_id_1 = "11111111111111111111111111111111"
    trace_id_2 = "22222222222222222222222222222222"

    # Verify decision is deterministic for the same trace_id
    res_1a = engine._should_sample(trace_id_1)
    res_1b = engine._should_sample(trace_id_1)
    assert res_1a == res_1b

    res_2a = engine._should_sample(trace_id_2)
    res_2b = engine._should_sample(trace_id_2)
    assert res_2a == res_2b


# ==========================================
# Upgrade 2: Four Golden Signals Tests
# ==========================================

def test_telemetry_four_golden_signals():
    engine = TelemetryEngine()
    
    # Check initial signals state
    signals = engine.get_golden_signals()
    assert signals["pipeline_run_count"] == 0
    assert signals["pipeline_error_count"] == 0
    assert signals["active_pipelines"] == 0
    assert signals["active_pipelines_peak"] == 0

    # Start pipeline
    engine.record_pipeline_start()
    signals = engine.get_golden_signals()
    assert signals["active_pipelines"] == 1
    assert signals["active_pipelines_peak"] == 1

    # Start another parallel pipeline (saturation)
    engine.record_pipeline_start()
    signals = engine.get_golden_signals()
    assert signals["active_pipelines"] == 2
    assert signals["active_pipelines_peak"] == 2

    # End first pipeline (success)
    engine.record_pipeline_end(had_error=False)
    signals = engine.get_golden_signals()
    assert signals["active_pipelines"] == 1
    assert signals["pipeline_run_count"] == 1
    assert signals["pipeline_error_count"] == 0

    # End second pipeline (error)
    engine.record_pipeline_end(had_error=True)
    signals = engine.get_golden_signals()
    assert signals["active_pipelines"] == 0
    assert signals["pipeline_run_count"] == 2
    assert signals["pipeline_error_count"] == 1
    assert signals["active_pipelines_peak"] == 2  # Peak holds high-water mark


def test_telemetry_four_golden_signals_thread_safety():
    engine = TelemetryEngine()
    num_threads = 10
    iterations = 50

    def worker():
        for _ in range(iterations):
            engine.record_pipeline_start()
            time.sleep(0.001)
            engine.record_pipeline_end(had_error=False)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    signals = engine.get_golden_signals()
    assert signals["pipeline_run_count"] == num_threads * iterations
    assert signals["active_pipelines"] == 0
    assert signals["active_pipelines_peak"] > 0


# ==========================================
# Upgrade 3: OpenTelemetry Semantic Convention
# ==========================================

def test_telemetry_otel_resource_block(tmp_path):
    custom_settings = Settings()
    log_file = tmp_path / "telemetry_otel.jsonl"
    custom_settings.telemetry_log_path = str(log_file)

    engine = TelemetryEngine(settings=custom_settings)
    engine.emit(event_type="otel_test")

    parsed = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert "resource" in parsed
    assert parsed["resource"]["service.name"] == "barrikade"
    assert parsed["resource"]["telemetry.sdk.language"] == "python"
    assert "service.version" in parsed["resource"]


def test_telemetry_w3c_trace_span_normalization():
    engine = TelemetryEngine()

    # Valid hex strings - should pad/truncate correctly
    # 1. Trace ID padding
    assert engine._normalize_trace_id("4a3b2c") == "000000000000000000000000004a3b2c"
    # 2. Trace ID truncating (keeps first 32 chars of normalized hex)
    long_trace = "a" * 40
    assert engine._normalize_trace_id(long_trace) == "a" * 32
    # 3. Span ID padding
    assert engine._normalize_span_id("1b2c") == "0000000000001b2c"
    # 4. Span ID truncating (keeps first 16 chars)
    long_span = "b" * 20
    assert engine._normalize_span_id(long_span) == "b" * 16

    # Legacy/Arbitrary non-hex strings - should remain unchanged for backward compatibility
    assert engine._normalize_trace_id("trace-abc") == "trace-abc"
    assert engine._normalize_span_id("span-xyz") == "span-xyz"


# ==========================================
# Upgrade 4: Cross-Session & Identity Correlation
# ==========================================

def test_session_identity_correlation(tmp_path):
    # Set up telemetry
    custom_settings = Settings()
    log_file = tmp_path / "telemetry_identity.jsonl"
    custom_settings.telemetry_log_path = str(log_file)
    custom_settings.telemetry_enabled = True

    # Patch settings and init orchestrator
    import core.session_orchestrator
    from unittest.mock import patch

    with patch("core.session_orchestrator.telemetry._settings", custom_settings):
        # We mock dependencies to isolate session lifecycle
        mock_pipeline = MagicMock()
        
        # Configure mocked pipeline detect return
        mock_pipeline_res = PipelineResult(
            input_hash="hash-123",
            total_processing_time_ms=10.0,
            layer_a_result={},
            layer_a_time_ms=5.0,
            layer_b_result=None,
            layer_b_time_ms=None,
            layer_c_result=None,
            layer_c_time_ms=None,
            layer_d_result=None,
            layer_d_time_ms=None,
            layer_e_result=None,
            layer_e_time_ms=None,
            final_verdict=FinalVerdict.ALLOW,
            decision_layer=DecisionLayer.LAYER_A,
            confidence_score=1.0,
        )
        mock_pipeline.detect.return_value = mock_pipeline_res

        class DummyRiskLevel:
            value = "low"

        class DummyDrift:
            drift_score = 0.05
            risk_level = DummyRiskLevel()
            def to_dict(self):
                return {"drift_score": 0.05, "risk_level": "low"}

        mock_scorer = MagicMock()
        mock_scorer.embed_intent.return_value = None
        mock_scorer.compute_drift.return_value = DummyDrift()

        mock_budget = MagicMock()

        class DummyReport:
            is_near_miss = False
            max_intent_drift_score = 0.05
            risk_budget_final = 10
            risk_budget_initial = 10

        mock_reporter = MagicMock()
        mock_reporter.generate_report.return_value = DummyReport()

        store = InMemorySessionStore()
        orchestrator = SessionOrchestrator(
            pipeline=mock_pipeline,
            session_store=store,
            intent_scorer=mock_scorer,
            risk_budget_engine=mock_budget,
            incident_reporter=mock_reporter,
        )

        # Start session with client identity context
        session_id = orchestrator.start_session(
            declared_intent="Access financial reports",
            permissions=["read_reports"],
            client_id="user-456",
            tenant_id="tenant-gold",
        )

        # Verify details stored on session
        session = store.get_session(session_id)
        assert session.client_id == "user-456"
        assert session.tenant_id == "tenant-gold"

        summary = orchestrator.get_session_summary(session_id)
        assert summary["client_id"] == "user-456"
        assert summary["tenant_id"] == "tenant-gold"

        # Detect with session context
        orchestrator.detect_with_session(session_id, "Get Gold Reports")

        # Close session
        orchestrator.end_session(session_id)

    # Verify telemetry logs written to file
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").splitlines()
    
    # We should have session_start, drift_check, and session_end events
    assert len(lines) >= 3
    
    start_event = json.loads(lines[0])
    assert start_event["event_type"] == "session_start"
    assert start_event["client_id"] == "user-456"
    assert start_event["tenant_id"] == "tenant-gold"

    drift_event = json.loads(lines[1])
    assert drift_event["event_type"] == "drift_check"
    assert drift_event["client_id"] == "user-456"
    assert drift_event["tenant_id"] == "tenant-gold"

    end_event = json.loads(lines[-1])
    assert end_event["event_type"] == "session_end"
    assert end_event["client_id"] == "user-456"
    assert end_event["tenant_id"] == "tenant-gold"
