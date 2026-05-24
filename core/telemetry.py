import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from core.__version__ import __version__ as barrikade_version

class TelemetryEngine:
    """A clean, thread-safe JSONL logging engine for Centralized Telemetry & Audit in Barrikade."""

    def __init__(self, settings: Optional[Any] = None) -> None:
        self._settings = settings
        self._logger = logging.getLogger("barrikade.telemetry")

    @property
    def settings(self) -> Any:
        if self._settings is None:
            from core.settings import Settings
            self._settings = Settings()
        return self._settings

    def emit(
        self,
        event_type: str,
        workload_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
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

        # Prepare compact, single-line event dict
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "workload_id": workload_id,
            "trace_id": trace_id,
            "span_id": span_id,
            "barrikade_version": barrikade_version,
            "payload": payload or {},
            "metrics": metrics or {},
        }

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
