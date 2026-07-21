"""Credential broker tool — the Registry pillar's ``request_credentials``.

Registered beside ``detect_prompt_injection`` on the same MCP server (via
``register(mcp)``, called from server.py — no second server, no import cycle).
The gate is deterministic (human∩agent policy intersection); every grant mints
a store-issued token via the ``SecretBackend``; every decision is audited with
the token's lease id, never the credential itself. All broker state is built
lazily on first call so a detection-only deployment (no broker env vars) still
gets a working server.
"""

import logging
import os
import threading
import time
import uuid

import anyio
from pydantic import BaseModel, Field

from barrikade_mcp.policy import is_allowed, load_policy
from barrikade_mcp.secret_backend import OpenBaoBackend


audit = logging.getLogger("barrikade_mcp.audit")
audit.setLevel(logging.INFO)
audit.propagate = False  # file only — never toward stdout (stdio transport owns it)


class CredentialResult(BaseModel):
    granted: bool = Field(description="Did you get it?")

    # only when granted -> optional
    token: str | None = Field(default=None, description="The scoped credential itself (Option A).")

    # only when granted -> optional
    expires_at: float | None = Field(
        default=None,
        description="So the agent (and audit log) know the limits of what it got. "
        "(e.g. when this token stops working)",
    )

    # only when granted -> optional
    scope: str | None = Field(default=None, description="What the token is allowed to do")

    # only when denied -> optional
    denied_reason: str | None = Field(default=None, description="Why not, when refused")


class _BrokerState:
    """Everything the tool needs, built once on first call — never at import."""

    def __init__(self):
        self.agent_id = os.environ.get("BARRIKADE_AGENT_ID")
        if not self.agent_id:
            raise RuntimeError("BARRIKADE_AGENT_ID is not set")
        audit_path = os.environ.get("BARRIKADE_AUDIT_PATH")
        if not audit_path:
            raise RuntimeError("BARRIKADE_AUDIT_PATH is not set")
        self.policy = load_policy()  # itself env-checked
        self.backend = OpenBaoBackend()  # itself env-checked
        self.ttl_s = int(os.environ.get("BARRIKADE_TOKEN_TTL_S", "600"))
        handler = logging.FileHandler(audit_path)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        formatter.converter = time.gmtime  # UTC, as before
        handler.setFormatter(formatter)
        audit.addHandler(handler)  # you know why this line matters


_state: _BrokerState | None = None
_state_lock = threading.Lock()


def _get_state() -> _BrokerState:
    """Build broker state lazily and thread-safely on first use.

    Same double-checked lock as ``server._get_pipeline``: a failed construction
    leaves the cache empty so the next call retries with fixed config.
    """
    global _state
    if _state is None:
        with _state_lock:
            if _state is None:
                _state = _BrokerState()
    return _state


def _request_sync(resource: str, action: str, reason: str) -> CredentialResult:
    """Blocking body; run in a worker thread off the event loop."""
    st = _get_state()
    trace = uuid.uuid4().hex[:8]
    scope, decision_reason = is_allowed(st.policy, st.agent_id, resource, action)

    if scope:
        token = st.backend.mint(scope, st.ttl_s)
        audit.info(
            "GRANT trace=%s policy_v=%s agent=%s resource=%s action=%s scope=%s "
            "lease=%s stated_reason=%r",
            trace,
            st.policy.get("version"),
            st.agent_id,
            resource,
            action,
            scope,
            token.lease_id,  # the ID, never the credential
            reason or "<none>",
        )
        return CredentialResult(
            granted=True, scope=scope, expires_at=token.expires_at, token=token.credential
        )

    audit.warning(
        "DENY trace=%s policy_v=%s agent=%s resource=%s action=%s stated_reason=%r decision=%s",
        trace,
        st.policy.get("version"),
        st.agent_id,
        resource,
        action,
        reason or "<none>",
        decision_reason,
    )
    return CredentialResult(granted=False, denied_reason=decision_reason)


def register(mcp) -> None:
    """Called by server.py — registers request_credentials on the shared server."""

    @mcp.tool()
    async def request_credentials(
        resource: str,
        action: str,
        *,
        reason: str,
    ) -> CredentialResult:
        """Request a scoped, short-lived credential for a resource/action.

        The broker grants only when BOTH the agent's policy scope AND its
        human owner's permissions cover the action (a deterministic check —
        no model in the loop), then mints a store-issued token with an
        enforced TTL. ``reason`` is required and audited: every grant and
        denial is recorded with a trace id and the token's lease id, never
        the credential itself.
        """
        return await anyio.to_thread.run_sync(_request_sync, resource, action, reason)
