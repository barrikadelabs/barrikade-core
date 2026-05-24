import json
import logging
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.settings import Settings
from core.telemetry import TelemetryEngine, telemetry
from core.session import InMemorySessionStore
from core.session_orchestrator import SessionOrchestrator
from models.verdicts import InputProvenance, FinalVerdict, DecisionLayer
from models.PipelineResult import PipelineResult


# ==========================================
# Core Telemetry Tests
# ==========================================

@pytest.mark.telemetry
def test_telemetry_singleton_export():
    assert isinstance(telemetry, TelemetryEngine)
    assert telemetry.settings is not None


@pytest.mark.telemetry
def test_telemetry_emit_success(tmp_path, caplog):
    # Set up custom settings pointing to the temp directory
    custom_settings = Settings()
    log_file = tmp_path / "telemetry.jsonl"
    custom_settings.telemetry_log_path = str(log_file)
    custom_settings.telemetry_enabled = True

    engine = TelemetryEngine(settings=custom_settings)

    with caplog.at_level(logging.INFO, logger="barrikade.telemetry"):
        engine.emit(
            event_type="test_event",
            workload_id="workload-123",
            trace_id="trace-abc",
            span_id="span-xyz",
            payload={"key": "value"},
            metrics={"latency_ms": 12.5},
        )

    # Check logger output
    assert len(caplog.records) == 1
    log_record = caplog.records[0]
    assert log_record.name == "barrikade.telemetry"
    
    parsed_log = json.loads(log_record.message)
    assert parsed_log["event_type"] == "test_event"
    assert parsed_log["workload_id"] == "workload-123"
    assert parsed_log["trace_id"] == "trace-abc"
    assert parsed_log["span_id"] == "span-xyz"
    assert parsed_log["payload"] == {"key": "value"}
    assert parsed_log["metrics"] == {"latency_ms": 12.5}
    assert "timestamp" in parsed_log
    assert "barrikade_version" in parsed_log

    # Check file output
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    
    parsed_file = json.loads(lines[0])
    assert parsed_file["event_type"] == "test_event"
    assert parsed_file["workload_id"] == "workload-123"
    assert parsed_file["trace_id"] == "trace-abc"
    assert parsed_file["span_id"] == "span-xyz"
    assert parsed_file["payload"] == {"key": "value"}
    assert parsed_file["metrics"] == {"latency_ms": 12.5}
    assert parsed_file["timestamp"] == parsed_log["timestamp"]


@pytest.mark.telemetry
def test_telemetry_disabled(tmp_path, caplog):
    custom_settings = Settings()
    log_file = tmp_path / "telemetry_disabled.jsonl"
    custom_settings.telemetry_log_path = str(log_file)
    custom_settings.telemetry_enabled = False

    engine = TelemetryEngine(settings=custom_settings)

    with caplog.at_level(logging.INFO, logger="barrikade.telemetry"):
        engine.emit(
            event_type="disabled_event",
            payload={"should": "not_log"},
        )

    # Logger shouldn't receive anything
    assert len(caplog.records) == 0
    # File shouldn't be created
    assert not log_file.exists()


@pytest.mark.telemetry
def test_telemetry_empty_payload_metrics(tmp_path):
    custom_settings = Settings()
    log_file = tmp_path / "telemetry_empty.jsonl"
    custom_settings.telemetry_log_path = str(log_file)

    engine = TelemetryEngine(settings=custom_settings)
    engine.emit(event_type="empty_event")

    assert log_file.exists()
    parsed = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert parsed["payload"] == {}
    assert parsed["metrics"] == {}
    assert parsed["workload_id"] is None
    assert parsed["trace_id"] is None
    assert parsed["span_id"] is None


@pytest.mark.telemetry
def test_telemetry_write_failure_graceful(tmp_path, caplog):
    custom_settings = Settings()
    # Point to a directory that cannot be created or written to
    invalid_dir = tmp_path / "readonly_dir"
    invalid_dir.mkdir()
    # Make directory read-only
    invalid_dir.chmod(0o400)
    
    log_file = invalid_dir / "subdir" / "telemetry.jsonl"
    custom_settings.telemetry_log_path = str(log_file)

    engine = TelemetryEngine(settings=custom_settings)

    # Ensure no exception is raised and a warning is logged
    with caplog.at_level(logging.WARNING, logger="barrikade.telemetry"):
        engine.emit(event_type="test_graceful")

    # Clean up chmod so pytest can delete tmp_path
    invalid_dir.chmod(0o700)

    # Check that a warning was indeed logged
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) >= 1
    assert "Failed to write" in warnings[0].message


# ==========================================
# Upgrade 1: Sampling Tests
# ==========================================

@pytest.mark.telemetry
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


@pytest.mark.telemetry
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


@pytest.mark.telemetry
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


@pytest.mark.telemetry
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

@pytest.mark.telemetry
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


@pytest.mark.telemetry
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

@pytest.mark.telemetry
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


@pytest.mark.telemetry
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

@pytest.mark.telemetry
def test_session_identity_correlation(tmp_path):
    # Set up telemetry
    custom_settings = Settings()
    log_file = tmp_path / "telemetry_identity.jsonl"
    custom_settings.telemetry_log_path = str(log_file)
    custom_settings.telemetry_enabled = True

    # Patch settings and init orchestrator
    import core.session_orchestrator

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
