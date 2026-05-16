# Barrikada Agentic Security Layer

Barrikada has evolved from a stateless per-request prompt-injection detector into a **stateful agentic security platform**. This document describes the new architecture, components, and how to use the session-aware API.

## Overview

Autonomous AI agents often execute complex workloads involving multiple tool calls, data source interactions, and external domain communications over a period of time. A stateless per-request detection model is insufficient to track longitudinal threats like *intent drift* or *privilege escalation*.

The Agentic Security Layer introduces stateful tracking across the lifecycle of a single agent task. It provides:
- **Workload Session Tracking**: Persistent records of declared intents, tool calls, permissions, external contacts, and data provenance.
- **Intent Deviation Scoring**: Longitudinal drift detection to measure how far an agent strays from its originally declared intent.
- **Risk Budget Engine**: A stateful counter that deducts "risky actions" (e.g., domain contacts, permission escalations) and halts execution if the budget is exhausted.
- **Incident & Near-Miss Reporting**: Standardized JSON reports containing the entire session context, enabling compliance audits and near-miss analysis.

All new components are designed to be strictly **additive**. The core pipeline (`PIPipeline.detect()`) and the stateless `/v1/detect` endpoint remain fully backward-compatible and unchanged.

---

## Architecture Components

The agentic extensions are orchestrated by the `SessionOrchestrator` (`core/session_orchestrator.py`), which composes the stateless `PIPipeline` with four new modules:

### 1. Workload Session Tracker (`core/session.py`)
Persists `WorkloadSession` objects containing a structured event log of the agent's actions. 
- Designed with an abstract `SessionStoreBackend` interface.
- Includes a thread-safe `InMemorySessionStore` with TTL-based eviction for easy out-of-the-box local usage (with TODOs explicitly marking where Redis/Postgres adapters should be implemented for production deployment).

### 2. Intent Deviation Scorer (`core/intent_scorer.py`)
Evaluates how closely subsequent actions match the session's initially declared intent.
- Uses `all-MiniLM-L6-v2` as a dedicated general-purpose embedding model instead of the fine-tuned Layer B model, ensuring the embedding space faithfully represents semantic closeness rather than attack/benign discrimination.
- The model is loaded lazily only when session features are used, preserving the performance of the stateless pipeline.

### 3. Risk Budget Engine (`core/risk_budget.py`)
Assigns a "risk budget" to each session. Risky actions (e.g., new external domains, pipeline flags) deduct from this budget based on configurable costs.
- Triggers an `ESCALATE` intervention if the budget is exhausted.
- Flips the session's status to `PAUSED` when this happens. The orchestrator rejects further `detect_with_session` calls on a non-`ACTIVE` session (HTTP layer maps to `409 Conflict`), so a single exhausted session cannot keep deducting budget on subsequent calls — human review is required before proceeding.
- Highly configurable via `SessionSettings`, allowing budget overrides per-session or cost tweaking per-category.

### 4. Incident Reporter (`core/incident_reporter.py`)
Compiles a standard `IncidentReport` (adhering to strict Pydantic schemas) when a session ends.
- Evaluates the session history to detect **near misses** (e.g., when a threat is successfully contained by a pipeline block or budget exhaustion without causing actual harm).
- Exposes full provenance, delegation chains, and tool invocations.

---

## New Interventions vs. Pipeline Verdicts

To maintain backward compatibility, the pipeline's `FinalVerdict` schema (`ALLOW`, `FLAG`, `BLOCK`) is completely distinct from session-level actions.

Session-level actions use the **`Intervention`** enum (`models/verdicts.py`):
- `NONE`: No session-level action required.
- `HALT`: Stop execution entirely.
- `DOWNGRADE`: Route the agent to a less capable (safer) model.
- `RESAMPLE`: Rerun the decision from a clean checkpoint (stripping untrusted content).
- `ESCALATE`: Require human approval before proceeding.
- `REVOKE`: Strip permissions mid-session.

---

## Configuration

Agentic features are configured via `SessionSettings` (`core/session_settings.py`). Key configurations include:

- `default_risk_budget`: The base budget assigned to new sessions (default: 5).
- `category_costs`: A dictionary mapping `RiskCategory` values to budget deduction amounts (e.g., `irreversible_action`: 2).
- `intent_drift_warn_threshold` and `intent_drift_block_threshold`: Define the boundaries for `MODERATE`, `HIGH`, and `CRITICAL` drift.
- `session_ttl_seconds`: Expiration time for in-memory sessions to prevent memory leaks.

---

## API Endpoints

The API server (`api/server.py`) has been expanded with the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/sessions` | Create a new session. Supply `declared_intent`, `permissions`, and `provenance`. Returns `session_id`. |
| `POST` | `/v1/sessions/{session_id}/detect` | Run a session-aware detection. Supply `text` and optional `tool_name`, `target_domain`. Returns the standard pipeline result merged with intent drift and risk assessment details. |
| `GET` | `/v1/sessions/{session_id}` | Retrieve a lightweight summary of an active session. |
| `POST` | `/v1/sessions/{session_id}/end` | Close a session and immediately generate a full `IncidentReport`. |
| `GET` | `/v1/sessions/{session_id}/report` | Retrieve the `IncidentReport` for an already closed session. |

### Example Workflow

#### 1. Start a Session
```bash
curl -X POST http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "declared_intent": "Summarize the Q3 financial report",
    "permissions": ["read_files"],
    "provenance": "trusted_internal",
    "risk_budget": 5
  }'
# Response: {"session_id": "abc12345..."}
```

#### 2. Run a Detection
```bash
curl -X POST http://localhost:8000/v1/sessions/abc12345.../detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Read the revenue data from the database",
    "provenance": "trusted_internal",
    "tool_name": "read_database"
  }'
# Response includes pipeline_result, drift scores, risk_assessment, and required intervention.
```

#### 3. End Session & Retrieve Report
```bash
curl -X POST http://localhost:8000/v1/sessions/abc12345.../end
# Response: {"report": { ...full incident report JSON... }}
```
