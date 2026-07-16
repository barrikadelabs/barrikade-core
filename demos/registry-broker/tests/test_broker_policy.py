import pytest
from broker_policy import covers, is_allowed, load_policy


@pytest.fixture
def policy():
    return load_policy()  # the anchored default — works from any cwd


# --- the truth table, one test per row ---


def test_grant_when_both_cover(policy):
    assert is_allowed(policy, "notion-bugfinder", "github", "read") == ("repo:read", "ok")


def test_deny_agent_scope_does_not_cover(policy):
    # deny #3: agent has the resource, but its scope doesn't cover the action
    assert is_allowed(policy, "notion-bugfinder", "github", "write") == (
        None,
        "agent scope 'repo:read' does not cover 'write'",
    )


def test_deny_owner_scope_does_not_cover(policy):
    # deny #4 — THE fixture case: agent over-provisioned beyond its read-only owner
    assert is_allowed(policy, "deploy-bot", "github", "write") == (
        None,
        "owner scope 'repo:read' does not cover 'write'",
    )


def test_deny_agent_lacks_resource(policy):
    # deny #1: agent has no entry for the resource at all
    assert is_allowed(policy, "notion-bugfinder", "stripe", "read") == (
        None,
        "agent has no access to this resource",
    )


def test_deny_owner_lacks_resource():
    # deny #2 has no YAML fixture on purpose — a minimal inline policy covers it:
    # the agent holds github, but its owner has NO github entry at all.
    policy = {
        "agents": {
            "bot": {"owner": "o@x", "allowed": [{"resource": "github", "scope": "repo:read"}]}
        },
        "humans": {"o@x": {"allowed": [{"resource": "notion", "scope": "pages:read"}]}},
    }
    assert is_allowed(policy, "bot", "github", "read") == (
        None,
        "human owner lacks this resource",
    )


# --- the grammar's edge cases ---


def test_substring_actions_are_denied(policy):
    # the old `action in agent_scope` bug granted all of these
    for action in ("e", "rea", "o:r"):
        scope, reason = is_allowed(policy, "notion-bugfinder", "github", action)
        assert scope is None
        assert reason == f"agent scope 'repo:read' does not cover '{action}'"


def test_covers_exact_match_only():
    assert covers("repo:read", "read") is True
    assert covers("repo:write", "write") is True
    assert covers("repo:read", "e") is False
    assert covers("repo:read", "rea") is False
    assert covers("repo:read", "write") is False


def test_multi_scope_agent_returns_matching_member():
    # the old next() bug saw only the FIRST entry; the grant must return the MATCHING member
    policy = {
        "agents": {
            "bot": {
                "owner": "o@x",
                "allowed": [
                    {"resource": "github", "scope": "repo:read"},
                    {"resource": "github", "scope": "repo:write"},
                ],
            }
        },
        "humans": {
            "o@x": {
                "allowed": [
                    {"resource": "github", "scope": "repo:read"},
                    {"resource": "github", "scope": "repo:write"},
                ]
            }
        },
    }
    assert is_allowed(policy, "bot", "github", "write") == ("repo:write", "ok")


# --- load-time validation ---

BAD_YAML = """\
agents:
  bot:
    owner: o@x
    allowed:
      - { resource: github, scope: "SCOPE" }
humans:
  o@x:
    allowed:
      - { resource: github, scope: "repo:read" }
"""


def test_load_rejects_unknown_action(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text(BAD_YAML.replace("SCOPE", "repo:wrtie"))
    with pytest.raises(ValueError, match="unknown action 'wrtie'"):
        load_policy(p)


def test_load_rejects_scope_without_colon(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text(BAD_YAML.replace("SCOPE", "repo"))
    with pytest.raises(ValueError, match="exactly one ':'"):
        load_policy(p)
