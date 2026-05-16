import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.orchestrator import PIPipeline
from core.session import SessionNotActiveError
from core.session_orchestrator import SessionOrchestrator, create_session_orchestrator
from core.session_settings import SessionSettings
from core.settings import Settings
from models.verdicts import InputProvenance

log = logging.getLogger(__name__)

@dataclass
class AppState:
    pipeline: PIPipeline | None = None
    session_orchestrator: SessionOrchestrator | None = None
    startup_error: str | None = None


# Stateless detect request/response


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    include_diagnostics: bool = False


class DetectResponse(BaseModel):
    final_verdict: str
    decision_layer: str
    confidence_score: float
    total_processing_time_ms: float
    result: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    pipeline_initialized: bool
    session_orchestrator_initialized: bool = False
    details: str | None = None


# Session request/response models


class CreateSessionRequest(BaseModel):
    declared_intent: str = Field(..., min_length=1, max_length=10000)
    permissions: list[str] = Field(default_factory=list)
    provenance: str = "unknown"
    delegation_chain: list[str] = Field(default_factory=list)
    risk_budget: int | None = None


class CreateSessionResponse(BaseModel):
    session_id: str


class SessionDetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    provenance: str = "unknown"
    tool_name: str | None = None
    target_domain: str | None = None


class SessionDetectResponse(BaseModel):
    pipeline_result: dict[str, Any]
    session_id: str
    drift: dict[str, Any] | None = None
    risk_assessment: dict[str, Any] | None = None
    intervention: str = "none"


class SessionSummaryResponse(BaseModel):
    session_id: str
    declared_intent: str
    status: str
    created_at: str
    closed_at: str | None = None
    event_count: int
    permissions_granted: list[str]
    permissions_requested: list[str]
    external_domains_contacted: list[str]
    delegation_chain: list[str]
    risk_budget_initial: int
    risk_budget_remaining: int


class IncidentReportResponse(BaseModel):
    report: dict[str, Any]


# App lifecycle


state = AppState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state.pipeline = PIPipeline()
        state.startup_error = None
        log.info("Barrikada pipeline initialized")

        # Initialise the session orchestrator, reusing the pipeline
        try:
            session_settings = SessionSettings()
            state.session_orchestrator = create_session_orchestrator(
                settings=session_settings,
                pipeline=state.pipeline,
            )
            log.info("Barrikada session orchestrator initialized")
        except Exception as exc:
            log.warning(
                "Session orchestrator initialization failed (stateless "
                "endpoint still available): %s",
                exc,
            )
            state.session_orchestrator = None

    except Exception as exc:  # pragma: no cover
        state.pipeline = None
        state.session_orchestrator = None
        state.startup_error = str(exc)
        log.exception("Failed to initialize Barrikada pipeline")
    yield


app = FastAPI(
    title="Barrikade Detection API",
    version="0.1.0",
    description="Production API for the Barrikade detection pipeline.",
    lifespan=lifespan,
)


# Health endpoints


@app.get("/health/live", response_model=HealthResponse)
def live():
    return HealthResponse(status="alive")


@app.get("/health/ready", response_model=ReadinessResponse)
def ready():
    if state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(state.startup_error or "Pipeline not initialized"),
        )

    settings = Settings()

    return ReadinessResponse(
        status="ready",
        pipeline_initialized=True,
        session_orchestrator_initialized=state.session_orchestrator is not None,
        details=f"Layer E judge active ({settings.layer_e_judge_mode}).",
    )


# Stateless detect endpoint


@app.post("/v1/detect", response_model=DetectResponse)
def detect(payload: DetectRequest):
    if state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=state.startup_error or "Pipeline unavailable",
        )

    try:
        result = state.pipeline.detect(payload.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        log.exception("Detection request failed")
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc

    details = result.to_dict() if payload.include_diagnostics else None
    return DetectResponse(
        final_verdict=result.final_verdict.value,
        decision_layer=result.decision_layer.value,
        confidence_score=result.confidence_score,
        total_processing_time_ms=result.total_processing_time_ms,
        result=details,
    )


# Session-aware endpoints


def _require_session_orchestrator() -> SessionOrchestrator:
    if state.session_orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Session orchestrator not initialized",
        )
    return state.session_orchestrator


def _parse_provenance(value: str) -> InputProvenance:
    try:
        return InputProvenance(value)
    except ValueError:
        return InputProvenance.UNKNOWN


@app.post("/v1/sessions", response_model=CreateSessionResponse)
def create_session(payload: CreateSessionRequest):
    """Create a new workload session."""
    orch = _require_session_orchestrator()
    try:
        session_id = orch.start_session(
            declared_intent=payload.declared_intent,
            permissions=payload.permissions,
            provenance=_parse_provenance(payload.provenance),
            delegation_chain=payload.delegation_chain,
            risk_budget=payload.risk_budget,
        )
    except Exception as exc:
        log.exception("Session creation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return CreateSessionResponse(session_id=session_id)


@app.post(
    "/v1/sessions/{session_id}/detect",
    response_model=SessionDetectResponse,
)
def session_detect(session_id: str, payload: SessionDetectRequest):
    """Run a session-aware detection."""
    orch = _require_session_orchestrator()
    try:
        result = orch.detect_with_session(
            session_id=session_id,
            input_text=payload.text,
            provenance=_parse_provenance(payload.provenance),
            tool_name=payload.tool_name,
            target_domain=payload.target_domain,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except SessionNotActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Session detection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SessionDetectResponse(**result.to_dict())


@app.get(
    "/v1/sessions/{session_id}",
    response_model=SessionSummaryResponse,
)
def get_session(session_id: str):
    """Get session status and summary."""
    orch = _require_session_orchestrator()
    try:
        summary = orch.get_session_summary(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return SessionSummaryResponse(**summary)


@app.post(
    "/v1/sessions/{session_id}/end",
    response_model=IncidentReportResponse,
)
def end_session(session_id: str):
    """End a session and get the incident report."""
    orch = _require_session_orchestrator()
    try:
        report = orch.end_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as exc:
        log.exception("Session end failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IncidentReportResponse(report=report.model_dump(mode="json"))


@app.get(
    "/v1/sessions/{session_id}/report",
    response_model=IncidentReportResponse,
)
def get_session_report(session_id: str):
    """Get the incident report for a (completed) session."""
    orch = _require_session_orchestrator()
    try:
        report = orch._reporter.generate_report(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    except Exception as exc:
        log.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IncidentReportResponse(report=report.model_dump(mode="json"))
