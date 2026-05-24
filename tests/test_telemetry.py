import json
import logging
from pathlib import Path
import pytest
from core.telemetry import TelemetryEngine, telemetry
from core.settings import Settings


def test_telemetry_singleton_export():
    assert isinstance(telemetry, TelemetryEngine)
    assert telemetry.settings is not None


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


def test_telemetry_write_failure_graceful(tmp_path, caplog):
    custom_settings = Settings()
    # Point to a directory that cannot be created or written to
    # On mac/linux, /proc/invalid-path or a directory we make read-only
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
