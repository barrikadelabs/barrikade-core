# Centered Observability & Telemetry System

Barrikade features an enterprise-ready, production-grade observability and centralized telemetry layer designed to capture runtime pipeline traces, security anomalies, and stateful session lifecycles. 

Telemetry logs conform strictly to the **OpenTelemetry (OTel) Semantic Conventions** and standard **W3C Trace Context Propagation** formats. This allows Barrikade's logs to be ingested natively and parsed out-of-the-box by collectors like the OpenTelemetry Collector, FluentBit, Jaeger, Prometheus, Grafana, and modern SIEM tools.

---

## 1. Core Architecture and Schema

Barrikade telemetry is written as single-line structured JSON (JSONL) records to a local audit log file. Each log record includes common telemetry metadata, OpenTelemetry resources, dynamic payload contents, and low-cardinality SRE metric blocks.

### The Unified JSONL Schema

Every telemetry event emits the following standard structure:

```json
{
  "timestamp": "2026-05-24T12:00:00.000000+01:00",
  "event_type": "pipeline_run",
  "workload_id": "sess-a1b2c3d4",
  "trace_id": "00000000000000000000000000trace12",
  "span_id": "0000000000span12",
  "resource": {
    "service.name": "barrikade",
    "service.version": "0.1.0",
    "telemetry.sdk.language": "python"
  },
  "barrikade_version": "0.1.0",
  "payload": {},
  "metrics": {},
  "client_id": "optional-user-id",
  "tenant_id": "optional-tenant-id"
}
```

* **`timestamp`**: ISO 8601 representation of the event time (including millisecond precision and timezone offset).
* **`event_type`**: The specific category of telemetry event. Common values:
  - `pipeline_run`: Represents a single execution pass of the prompt-injection detection pipeline.
  - `session_start`: Fired when a stateful security session is initialized.
  - `drift_check`: Emitted on every intent drift comparison check.
  - `risk_budget_deduction`: Recorded when a stateful action deducts from the user's risk budget.
  - `intervention_triggered`: Triggered when an automated escalation (e.g. pausing, blocking, or escalating permissions) occurs.
  - `session_end`: Fired when a session is finalized and a summary report is generated.
* **`resource`**: The OpenTelemetry resource block. Pins the service name, version, and language for automatic service mapping in distributed tracing dashboards.
* **`client_id` / `tenant_id`**: Correlation identities injected during stateful sessions to group multiple sessions belonging to the same actor.

---

## 2. OpenTelemetry & W3C Trace Normalization

Barrikade dynamically handles distributed trace correlation using standard W3C standards:

1. **Trace ID Normalization**: All incoming `trace_id` values are padded or truncated to a standard **32-character hexadecimal** string.
   - Example: `"4a3b2c"` $\rightarrow$ `"000000000000000000000000004a3b2c"`
2. **Span ID Normalization**: All incoming `span_id` values are padded or truncated to a standard **16-character hexadecimal** string.
   - Example: `"1b2c"` $\rightarrow$ `"0000000000001b2c"`
3. **Backward Compatibility Fallback**: If an arbitrary or legacy string containing non-hex characters (e.g., `"trace-abc"`) is provided, Barrikade's normalizer bypasses padding and propagates it exactly as-is to prevent downstream string equality failures in mock tests and legacy integrations.

---

## 3. Tail-Based Deterministic Sampling

In high-throughput enterprise deployments, logging 100% of safe pipeline runs creates unnecessary log volume and high ingestion costs. Barrikade resolves this using **tail-based deterministic safe-sampling**:

> [!IMPORTANT]
> **No Safe-Sample Friction out-of-the-box:**  
> By default, `telemetry_safe_sample_rate` is set to `1.0` (100% logging). Out-of-the-box, no telemetry records are dropped. Safe-sampling is an opt-in escape hatch for high-traffic environments.

```
                  [ Pipeline Execution Result ]
                               |
            Is Anomalous? (Verdicts: BLOCK/FLAG)
                     /                   \
                   YES                   NO
                   /                       \
             [ Log 100% ]          Deterministically Hash Trace ID
                                           |
                                  Hash < Sample Rate?
                                     /           \
                                   YES           NO
                                   /               \
                             [ Log Event ]     [ Drop Event ]
```

### Sampling Guidelines:
1. **Anomalies are always logged at 100%**: Any pipeline execution resulting in a `BLOCK` or `FLAG` final verdict is considered an anomaly and bypasses sampling, regardless of the configured rate.
2. **Session control events are always logged at 100%**: Events like `session_start`, `session_end`, `intervention_triggered`, and `risk_budget_deduction` are critical for security audits and are never sampled.
3. **Deterministic Sampling Algorithm**: Safe runs (`ALLOW` verdict) are sampled deterministically based on their `trace_id`. The trace ID is hashed using MD5; the hash is compared against the `telemetry_safe_sample_rate` value (between `0.0` and `1.0`). This guarantees that all downstream services participating in the same trace make the exact same sampling decision.

---

## 4. Google SRE Four Golden Signals

Barrikade instruments all core engine actions with thread-safe telemetry counters. This allows engineering teams to construct standard SRE dashboard monitors mapping **Latency, Traffic, Errors, and Saturation**:

* **Latency**: Every `pipeline_run` tracks execution durations in milliseconds (total processing time plus per-layer timing: `layer_a_time_ms`, `layer_b_time_ms`, etc.).
* **Traffic**: Captured via `pipeline_run_count` which registers the total count of pipeline requests over the process lifecycle.
* **Errors**: Total framework-level or API failures are registered via `pipeline_error_count`. In addition, individual events capture granular information:
  - `error_count`: The count of specific layer errors caught during execution.
  - `layer_errors`: A detailed structured list showing exactly which layers encountered issues (e.g. `[{"layer": "E", "error": "LLM Timeout"}]`).
* **Saturation**: Monitored via concurrency tracking:
  - `active_pipelines`: Current count of pipeline threads executing concurrently.
  - `active_pipelines_peak`: High-water mark of concurrent runs, indicating resource pressure.

---

## 5. Cross-Session & Actor Identity Correlation

To identify advanced, slow-drip prompt-injection attacks that span across multiple sessions over time, Barrikade correlates telemetry using identity tagging:

- When initializing a stateful session via the `SessionOrchestrator`, developers can pass optional `client_id` (e.g., `user-9912`) and `tenant_id` (e.g., `tenant-enterprise`) values.
- These credentials persist inside the session.
- Barrikade dynamically appends these tags to every lifecycle audit log, allowing SIEM systems (Splunk, Elastic, Sentinel) to aggregate telemetry records and run behavioral analytics grouped by actor or organization.

---

## 6. Telemetry Event Examples

### A. Stateless Pipeline Run (`pipeline_run`) — Logged Safe Run (Sampled)
```json
{
  "timestamp": "2026-05-24T12:00:05.123456+01:00",
  "event_type": "pipeline_run",
  "workload_id": "sess-a1b2c3d4",
  "trace_id": "00000000000000000000000000trace12",
  "span_id": "0000000000span12",
  "resource": {
    "service.name": "barrikade",
    "service.version": "0.1.0",
    "telemetry.sdk.language": "python"
  },
  "barrikade_version": "0.1.0",
  "payload": {
    "input_hash": "a4f8b912",
    "final_verdict": "allow",
    "decision_layer": "B",
    "layer_a_verdict": "allow",
    "layer_b_verdict": "allow",
    "layer_errors": []
  },
  "metrics": {
    "total_processing_time_ms": 14.2,
    "risk_score": 0.0,
    "layer_a_time_ms": 6.1,
    "layer_b_time_ms": 8.1,
    "pipeline_run_count": 142,
    "pipeline_error_count": 0,
    "active_pipelines": 0,
    "active_pipelines_peak": 3,
    "sampled": true
  }
}
```

### B. Identity-Scoped Session Initialized (`session_start`)
```json
{
  "timestamp": "2026-05-24T12:00:00.012345+01:00",
  "event_type": "session_start",
  "workload_id": "sess-a1b2c3d4",
  "trace_id": "00000000000000000000000000trace12",
  "span_id": "0000000000span12",
  "resource": {
    "service.name": "barrikade",
    "service.version": "0.1.0",
    "telemetry.sdk.language": "python"
  },
  "barrikade_version": "0.1.0",
  "payload": {
    "declared_intent": "Access financial reports",
    "permissions": ["read_reports"],
    "provenance": "trusted_internal",
    "delegation_chain": []
  },
  "metrics": {
    "risk_budget_initial": 10
  },
  "client_id": "user-456",
  "tenant_id": "tenant-gold"
}
```

---

## 7. Developer Reference and Configuration

Settings are controlled dynamically via the standard `Settings` config object (environment variables prefix: `BARRIKADE_`):

| Setting Key | Environment Variable | Type | Default | Description |
|---|---|---|---|---|
| `telemetry_enabled` | `BARRIKADE_TELEMETRY_ENABLED` | `bool` | `True` | Global toggle to enable/disable telemetry logging entirely. |
| `telemetry_log_path` | `BARRIKADE_TELEMETRY_LOG_PATH` | `str` | `"test_results/barrikade_telemetry.jsonl"` | Local path to write single-line structure JSON log records. |
| `telemetry_safe_sample_rate` | `BARRIKADE_TELEMETRY_SAFE_SAMPLE_RATE` | `float` | `1.0` | Target rate of logged safe `allow` runs. Set between `0.0` (drop all safe runs) and `1.0` (log all runs). |

---

## 8. Verifying Telemetry Functionality

To run the telemetry verification and engine upgrades test suite, use the dedicated `telemetry` marker:

```bash
# Run isolated telemetry tests
venv/bin/pytest tests/telemetry/

# Or run by markers
venv/bin/pytest -m telemetry
```
