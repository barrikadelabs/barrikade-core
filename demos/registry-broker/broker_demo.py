"""
a "Server" exposing a "Tool" over the "stdio" transport
"""

import logging
import time
import uuid
from datetime import datetime, timezone  # Import datetime, timezone
from pathlib import Path

from broker_policy import is_allowed, load_policy
from mcp.server.fastmcp import FastMCP  # Import FastMCP
from pydantic import BaseModel, Field


# --- audit logger setup ---
audit = logging.getLogger("broker-audit")
audit.setLevel(logging.INFO)

_handler = logging.FileHandler(
    Path(__file__).parent / "audit.log"
)  # the file it persists to (append by default)
_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
_formatter.converter = time.gmtime  # audit lines in UTC, matching expires_at
_handler.setFormatter(_formatter)
audit.addHandler(_handler)

mcp = FastMCP("broker")  # names your server, the client sees "broker"


class CredentialResult(BaseModel):
    granted: bool = Field(description="Did you get it?")

    # only when granted → optional
    token: str | None = Field(default=None, description="The scoped credential itself (Option A).")

    # only when granted → optional
    expires_at: float | None = Field(
        default=None,
        description="So the agent (and audit log) know the limits of what it got. "
        "(e.g. when this token stops working)",
    )

    # only when granted → optional
    scope: str | None = Field(default=None, description="What the token is allowed to do")

    # only when denied → optional
    denied_reason: str | None = Field(default=None, description="Why not, when refused")


AGENT_ID = "notion-bugfinder"  # The agent this broker serves (NOT a tool arg) . Hardcoded
POLICY = load_policy()


@mcp.tool()
async def request_credentials(
    resource: str,  # Github, Stripe, Databse
    action: str,  # Read, Write, Query (for scoping + audit)
    *,  # Everything after thus must be passed by name
    reason: str,  # Agent's stated reasons (audit trail), keyword-only AND required - no default
) -> CredentialResult:
    trace = uuid.uuid4().hex[:8]
    scope, decision_reason = is_allowed(POLICY, AGENT_ID, resource, action)

    if scope:
        expires_at = (
            datetime.now(timezone.utc).timestamp() + 600  # TTL - time right now + 600s (10mins)
        )
        token = f"scoped-token::{resource}::{scope}::exp={expires_at}"
        audit.info(
            "GRANT trace=%s policy_v=%s agent=%s resource=%s action=%s scope=%s stated_reason=%r",
            trace,
            POLICY.get("version"),
            AGENT_ID,
            resource,
            action,
            scope,
            reason or "<none>",
        )  # requested action AND granted scope

        return CredentialResult(granted=True, scope=scope, expires_at=expires_at, token=token)

    else:
        audit.warning(
            "DENY trace=%s policy_v=%s agent=%s resource=%s action=%s stated_reason=%r decision=%s",
            trace,
            POLICY.get("version"),
            AGENT_ID,
            resource,
            action,
            reason or "<none>",
            decision_reason,
        )  # same column order as GRANT, decision last

        return CredentialResult(granted=False, denied_reason=decision_reason)


if __name__ == "__main__":
    mcp.run()  # starts the server on stdio transport


# async def main():
#     result = await request_credentials("github")  # allowed -> should grant
#     print(result)

#     result = await request_credentials("stripe") # not in policy -> should deny
#     print(result)

# if __name__ == "__main__":
#     asyncio.run(main())
