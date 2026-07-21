import os

import yaml


VALID_ACTIONS = {"read", "write"}


def load_policy(path: str | None = None):
    # explicit arg for tests; env var for deployment; never a guessed default
    path = path or os.environ.get("BARRIKADE_POLICY_PATH")
    if not path:
        raise RuntimeError("BARRIKADE_POLICY_PATH is not set (point it at your agent_policy.yaml)")
    with open(path) as f:
        policy = yaml.safe_load(f)

    for section in ("agents", "humans"):
        for name, entry in policy[section].items():
            for e in entry["allowed"]:
                scope = e["scope"]
                if scope.count(":") != 1:
                    raise ValueError(
                        f"invalid scope '{scope}' for {section}/{name}: "
                        f"expected exactly one ':' (<category>:<action>)"
                    )
                if scope.split(":")[1] not in VALID_ACTIONS:
                    raise ValueError(
                        f"invalid scope '{scope}' for {section}/{name}: "
                        f"unknown action '{scope.split(':')[1]}' (valid: {sorted(VALID_ACTIONS)})"
                    )

    return policy


def covers(scope: str, action: str) -> bool:
    # A scope covers an action iff the action segment matches exactly.
    # covers("repo:read", "read") -> True
    # covers("repo:read", "e")    -> False   (substring is NOT coverage)

    return scope.split(":")[1] == action


def scopes_for(allowed: list[dict], resource: str) -> list[str]:
    # ALL scopes a party holds for a resource - never just the first.

    return [e["scope"] for e in allowed if e["resource"] == resource]


def is_allowed(policy, agent, resource, action):
    # Step 1 — agent side: find the agent's entry for this resource
    agent_allowed = policy["agents"][agent]["allowed"]  # a list of {resource, scope} dicts
    agent_scopes = scopes_for(agent_allowed, resource)

    if not agent_scopes:
        return (
            None,
            "agent has no access to this resource",
        )  # agent isn't permitted this resource at all

    # Step 2 — human side: find the human's entry for this resource
    owner = policy["agents"][agent]["owner"]  
    human_allowed = policy["humans"][owner][
        "allowed"
    ]  # NOW it's the list of {resource, scope} dicts
    human_scopes = scopes_for(human_allowed, resource)

    if not human_scopes:
        return (
            None,
            "human owner lacks this resource",
        )  # human isn't permitted -> intersection empty

    # Step 3 — both sides had it: return the agent's (narrower) scope
    for s in agent_scopes:
        if covers(s, action):
            # key 2 — the OWNER must also cover the action (grammar Q4: both keys turn)
            if any(covers(h, action) for h in human_scopes):
                return s, "ok"
            return None, f"owner scope '{', '.join(human_scopes)}' does not cover '{action}'"
    return (
        None,
        f"agent scope '{', '.join(agent_scopes)}' does not cover '{action}'",
    )  # scope doesn't cover this action -> deny

