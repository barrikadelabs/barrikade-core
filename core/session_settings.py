"""Session settings for the agentic security modules."""

from pydantic import BaseModel


class SessionSettings(BaseModel):
    """Configuration for workload sessions, risk budgets, and intent drift.

    These settings are separate from ``core.settings.Settings`` to keep the
    agentic extensions cleanly isolated from the per-request pipeline config.
    """

    # ── Risk Budget ─────────────────────────────────────────────────────
    # Default total budget per workload session.  Can be overridden
    # per-session via ``start_session(risk_budget=...)``.
    default_risk_budget: int = 5

    # Per-category cost overrides.  Keys are ``RiskCategory`` values.
    # Categories not listed here default to cost 1.
    # Example: {"irreversible_action": 2, "new_external_domain": 0}
    category_costs: dict[str, int] = {
        "permission_expansion": 1,
        "new_external_domain": 1,
        "untrusted_to_trusted_data_flow": 1,
        "irreversible_action": 2,
        "high_intent_drift": 1,
        "pipeline_flag": 1,
    }

    # ── Intent Drift ────────────────────────────────────────────────────
    intent_drift_warn_threshold: float = 0.35
    intent_drift_block_threshold: float = 0.55

    # ── Session Lifecycle ───────────────────────────────────────────────
    # Time-to-live in seconds.  Sessions older than this are evicted on
    # the next access to prevent unbounded memory growth.
    session_ttl_seconds: int = 3600

    max_events_per_session: int = 10_000

    # ── Intent Embedding Model ──────────────────────────────────────────
    # Loaded lazily only when a session is created (does not affect the
    # stateless detect() hot path).
    intent_embedding_model: str = "all-MiniLM-L6-v2"
