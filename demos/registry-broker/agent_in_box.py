"""
a "Client" and the initialize() call is the "lifecycle handshake"
"""

# agent_in_box.py — simulates an agent trapped inside the sandbox
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


BROKER_URL = "http://127.0.0.1:8000/mcp"  # the ONLY way the agent reaches the broker


async def broker_credentials_call():
    async with streamable_http_client(BROKER_URL) as (read, write, _):  # open the wire
        async with ClientSession(read, write) as session:  # a session on it
            await session.initialize()  # the "hello" 

            result = await session.call_tool(
                "request_credentials",
                {
                    "resource": "github",  # the object acted on, e.g. "s3://bucket/key", "users/42"
                    "action": "read",  # the operation, e.g. "read", "delete", "write"
                    "reason": "need repo access",  # justification string for audit/approval
                },  # HOLE 1: the three args — resource / action / reason
            )
            print("[agent] raw result:", result)  # print once to SEE what comes back
            return result.structuredContent  # data out


SECRET_PATH = "/workspace/../demo-secret.env"  # nosec B105 — a demo file path, not a credential; the host secret lives OUTSIDE the box


async def main():
    # Route 1 - the direct grab: just read the secret off disk
    try:
        with open(SECRET_PATH) as f:
            print("[agent] got secret directly:", f.read().strip())
    except FileNotFoundError:
        # Blocked by the sandbox - fall through to the sanctioned route
        print("[agent] direct read BLOCKED - secret isn't in the box")

        # Route 2 — the escape hatch: ask the broker
        result = await broker_credentials_call() 
        print("[agent] broker says granted =", result["granted"])
        print("[agent] scoped token =", result.get("token"))


asyncio.run(main())
