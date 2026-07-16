# Registry credential broker — demo

An MCP `request_credentials` tool behind a deterministic human∩agent policy gate:
the agent runs in a sandbox where reading secrets directly fails, the broker is the
only sanctioned way out, and every grant/deny lands in a persistent audit log.

## Run

```bash
./demo_all.sh            # all four parts (Part 3 needs Docker Desktop running)
./demo_all.sh 2 3        # just the broker + sandbox loop
./demo_all.sh --fresh    # reset the audit log first (append-only by default)
```

## The four parts

1. **Policy engine** (`broker_policy.py`) — pure decision logic, human∩agent intersection over `agent_policy.yaml`.
2. **Broker** (`broker_demo.py`) — the MCP tool: policy decision → scoped, TTL'd token or a structured denial with its specific reason.
3. **Sandbox loop** (`agent_in_box.py` in Docker) — the agent's direct secret read fails; the broker is the only way out.
4. **Audit log** (`audit.log`, generated) — every decision above, timestamped and leveled (denials are WARNINGs).

Status: prototype — policy-gate hardening in progress, see commit history.
