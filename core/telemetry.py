import hashlib
import json
import logging
import random
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from core.__version__ import __version__ as barrikade_version

# Static OTel resource conventions
OTEL_RESOURCE = {
    "service.name": "barrikade",
    "service.version": barrikade_version,
    "telemetry.sdk.language": "python",
}

class TelemetryEngine:
    """A clean, thread-safe JSONL logging engine for Centralized Telemetry & Audit in Barrikade."""

    def __init__(self, settings: Optional[Any] = None) -> None:
        self._settings = settings
        self._logger = logging.getLogger("barrikade.telemetry")
        
        # Golden Signals threading counters
        self._counters_lock = threading.Lock()
        self._pipeline_run_count = 0        # Traffic signal
        self._pipeline_error_count = 0      # Errors signal
        self._active_pipelines = 0          # Saturation signal
        self._active_pipelines_peak = 0     # Peak saturation high-water mark

    @property
    def settings(self) -> Any:
        if self._settings is None:
            from core.settings import Settings
            self._settings = Settings()
        return self._settings

    def record_pipeline_start(self) -> None:
        """Track pipeline concurrency (saturation signal)."""
        with self._counters_lock:
            self._active_pipelines += 1
            if self._active_pipelines > self._active_pipelines_peak:
                self._active_pipelines_peak = self._active_pipelines

    def record_pipeline_end(self, had_error: bool = False) -> None:
        """Track pipeline completion and errors (Traffic and Errors signals)."""
        with self._counters_lock:
            self._active_pipelines = max(0, self._active_pipelines - 1)
            self._pipeline_run_count += 1
            if had_error:
                self._pipeline_error_count += 1

    def get_golden_signals(self) -> Dict[str, int]:
        """Snapshot of the current Four Golden Signals counters."""
        with self._counters_lock:
            return {
                "pipeline_run_count": self._pipeline_run_count,
                "pipeline_error_count": self._pipeline_error_count,
                "active_pipelines": self._active_pipelines,
                "active_pipelines_peak": self._active_pipelines_peak,
            }

    def _normalize_trace_id(self, trace_id: Optional[str]) -> Optional[str]:
        """Pad/truncate trace_id to W3C standard (32 hex characters) if it's hex, otherwise return as is."""
        if not trace_id:
            return None
        clean = trace_id.replace("-", "").lower()
        if all(c in "0123456789abcdef" for c in clean):
            return clean.zfill(32)[:32] if clean else None
        return trace_id

    def _normalize_span_id(self, span_id: Optional[str]) -> Optional[str]:
        """Pad/truncate span_id to W3C standard (16 hex characters) if it's hex, otherwise return as is."""
        if not span_id:
            return None
        clean = span_id.replace("-", "").lower()
        if all(c in "0123456789abcdef" for c in clean):
            return clean.zfill(16)[:16] if clean else None
        return span_id

    def _should_sample(self, trace_id: Optional[str]) -> bool:
        """Determine if a safe event should be logged based on sample rate."""
        try:
            rate = self.settings.telemetry_safe_sample_rate
        except Exception:
            rate = 1.0
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False

        # Deterministic sampling based on trace_id for consistency
        norm_trace = self._normalize_trace_id(trace_id)
        if norm_trace:
            hash_val = int(hashlib.md5(norm_trace.encode(), usedforsecurity=False).hexdigest()[:8], 16)
            return (hash_val / 0xFFFFFFFF) < rate

        return random.random() < rate  # nosec B311

    def emit_sampled(
        self,
        *,
        is_anomalous: bool = False,
        force_log: bool = False,
        **kwargs,
    ) -> None:
        """Emit with tail-based sampling. Anomalous events are always logged."""
        if is_anomalous or force_log:
            self.emit(**kwargs)
        elif self._should_sample(kwargs.get("trace_id")):
            # Inject sampled indicator metric
            metrics = kwargs.setdefault("metrics", {})
            metrics["sampled"] = True
            self.emit(**kwargs)

    def emit(
        self,
        event_type: str,
        workload_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Format and log telemetry events as a single-line JSON.

        The event is logged to the 'barrikade.telemetry' python logger and,
        if enabled, appended directly to the configured telemetry JSONL file.
        """
        try:
            enabled = self.settings.telemetry_enabled
        except Exception:
            enabled = True

        if not enabled:
            return

        norm_trace = self._normalize_trace_id(trace_id)
        norm_span = self._normalize_span_id(span_id)

        # Prepare compact, single-line event dict matching OTel resource convention
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "workload_id": workload_id,
            "trace_id": norm_trace,
            "span_id": norm_span,
            "resource": OTEL_RESOURCE,
            "barrikade_version": barrikade_version,
            "payload": payload or {},
            "metrics": metrics or {},
        }

        # Optional client identity context fields (Upgrade 4)
        if client_id is not None:
            event["client_id"] = client_id
        if tenant_id is not None:
            event["tenant_id"] = tenant_id

        # Convert to compact single-line JSON string
        try:
            json_line = json.dumps(event, separators=(",", ":"))
        except Exception as e:
            self._logger.warning("Telemetry event serialization failed: %s", e)
            return

        # 1. Log to the dedicated python logger
        self._logger.info(json_line)

        # 2. Append directly to the telemetry log file
        try:
            log_path_str = self.settings.telemetry_log_path
        except Exception:
            log_path_str = "test_results/barrikade_telemetry.jsonl"

        if not log_path_str:
            return

        log_path = Path(log_path_str)
        try:
            # Handle directory creation gracefully
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Append directly to the file
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
        except Exception as e:
            # Handle write or creation failures gracefully
            self._logger.warning("Failed to write to telemetry log path '%s': %s", log_path, e)


# Singleton instance exported as standard
telemetry = TelemetryEngine()
