"""Broker tool tests — mocked backend, tmp policy, no OpenBao needed."""

import asyncio

import pytest
from barrikade_mcp import broker
from barrikade_mcp.secret_backend import ScopedToken


POLICY_YAML = """\
version: 1
agents:
  test-agent:
    owner: tester@x
    allowed:
      - { resource: github, scope: "repo:read" }
  over-bot:
    owner: readonly@x
    allowed:
      - { resource: github, scope: "repo:write" }
humans:
  tester@x:
    allowed:
      - { resource: github, scope: "repo:read" }
      - { resource: github, scope: "repo:write" }
  readonly@x:
    allowed:
      - { resource: github, scope: "repo:read" }
"""


class _FakeBackend:
    """Stands in for OpenBao: mints predictably, records revokes."""

    capability_grade = "hard-lease"

    def __init__(self):
        self.revoked = []

    def mint(self, scope, ttl_seconds):
        return ScopedToken(credential="fake-secret-token", lease_id="lease-abc123", expires_at=1e12)

    def revoke(self, lease_id):
        self.revoked.append(lease_id)


@pytest.fixture(autouse=True)
def _reset_broker_state():
    """The broker caches state in a module global — every test starts fresh."""
    broker._state = None
    broker.audit.handlers.clear()  # else each state-build adds another file handler
    yield
    broker._state = None
    broker.audit.handlers.clear()


@pytest.fixture
def broker_env(tmp_path, monkeypatch):
    """Full working env: tmp policy + tmp audit path + fake backend. Returns audit path."""
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(POLICY_YAML)
    audit_path = tmp_path / "audit.log"
    monkeypatch.setenv("BARRIKADE_POLICY_PATH", str(policy_path))
    monkeypatch.setenv("BARRIKADE_AUDIT_PATH", str(audit_path))
    monkeypatch.setenv("BARRIKADE_AGENT_ID", "test-agent")
    monkeypatch.setenv("OPENBAO_TOKEN", "unused-in-tests")
    monkeypatch.setattr(broker, "OpenBaoBackend", _FakeBackend)  # no network, ever
    return audit_path


def test_grant_returns_minted_token(broker_env):
    result = broker._request_sync("github", "read", "testing")
    assert result.granted is True
    assert result.token == "fake-secret-token"
    assert result.scope == "repo:read"
    assert result.denied_reason is None


def test_deny_agent_scope_does_not_cover(broker_env):
    result = broker._request_sync("github", "write", "testing")
    assert result.granted is False
    assert result.token is None
    assert result.denied_reason == "agent scope 'repo:read' does not cover 'write'"


def test_deny_owner_scope_does_not_cover(broker_env, monkeypatch):
    # switch the served agent BEFORE the first call — state builds and caches on first use
    monkeypatch.setenv("BARRIKADE_AGENT_ID", "over-bot")
    result = broker._request_sync("github", "write", "testing")
    assert result.granted is False
    assert result.denied_reason == "owner scope 'repo:read' does not cover 'write'"


def test_missing_env_fails_loudly_then_recovers(broker_env, monkeypatch):
    monkeypatch.delenv("BARRIKADE_AGENT_ID")
    with pytest.raises(RuntimeError, match="BARRIKADE_AGENT_ID"):
        broker._request_sync("github", "read", "testing")

    # failed construction left the cache empty — fixing the config makes the SAME call work
    monkeypatch.setenv("BARRIKADE_AGENT_ID", "test-agent")
    result = broker._request_sync("github", "read", "testing")
    assert result.granted is True


def test_audit_logs_lease_id_never_the_credential(broker_env):
    broker._request_sync("github", "read", "audit me")
    text = broker_env.read_text()
    assert "lease-abc123" in text
    assert "fake-secret-token" not in text


def test_both_tools_registered_without_broker_env():
    # the Rung-15 acceptance, frozen as a test: importing the server needs NO broker env
    from barrikade_mcp.server import mcp  # noqa: PLC0415 — lazy import is the point of the test

    names = [t.name for t in asyncio.run(mcp.list_tools())]
    assert names == ["detect_prompt_injection", "request_credentials"]
