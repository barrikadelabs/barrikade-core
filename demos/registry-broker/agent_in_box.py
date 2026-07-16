
# agent_in_box.py — simulates an agent trapped inside the sandbox
import asyncio
import broker_demo

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
        result = await broker_credentials_call()  # fill this in
        print("[agent] broker says granted =", result.granted)
        print("[agent] scoped token =", result.token)




async def broker_credentials_call():
    return await broker_demo.request_credentials("github", "read", reason="need repo access")

asyncio.run(main())