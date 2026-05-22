"""Public SDK for Barrikade."""

import logging
import os

from core.__version__ import __version__
from barrikade.sdk import (
    PIPipeline,
    SessionOrchestrator,
    create_session_orchestrator,
    SessionDetectResult,
    SessionSettings,
    SessionEvent,
    SessionEventType,
    SessionNotActiveError,
    SessionStatus,
    WorkloadSession,
    SessionStoreBackend,
    InMemorySessionStore,
    InputProvenance,
    Intervention,
    IncidentReport,
)
from core.artifacts import (
    ArtifactDownloadError,
    download_runtime_bundle,
    download_runtime_artifacts,
    ensure_runtime_bundle,
    ensure_runtime_artifacts,
)

_SDK_LOGGING_READY = False


def _ensure_sdk_logging() -> None:
    global _SDK_LOGGING_READY
    if _SDK_LOGGING_READY:
        return

    root_logger = logging.getLogger()
    if root_logger.handlers:
        _SDK_LOGGING_READY = True
        return

    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    for name in ("barrikade", "core"):
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    _SDK_LOGGING_READY = True


_ensure_sdk_logging()

if os.getenv("BARRIKADA_SKIP_IMPORT_BUNDLE_CHECK", "0") == "0":
    ensure_runtime_bundle()

__all__ = [
    "__version__",
    "ArtifactDownloadError",
    "PIPipeline",
    "download_runtime_bundle",
    "download_runtime_artifacts",
    "ensure_runtime_bundle",
    "ensure_runtime_artifacts",
    "SessionOrchestrator",
    "create_session_orchestrator",
    "SessionDetectResult",
    "SessionSettings",
    "SessionEvent",
    "SessionEventType",
    "SessionNotActiveError",
    "SessionStatus",
    "WorkloadSession",
    "SessionStoreBackend",
    "InMemorySessionStore",
    "InputProvenance",
    "Intervention",
    "IncidentReport",
]

