#!/usr/bin/env bash
# demo_all.sh — "show me everything" for the Registry-pillar prototype.
# Runs the whole stack end-to-end: policy engine -> broker -> sandbox loop -> audit log.
#
# Usage (from anywhere, in Git Bash):
#   ./demo_all.sh          run all four parts
#   ./demo_all.sh 2 3      run only Parts 2 and 3
#   ./demo_all.sh 1        run only Part 1
set -u
cd "$(dirname "$0")"                 # anchor to the script's own dir regardless of where it's called
export MSYS_NO_PATHCONV=1            # stop Git Bash mangling container-side /workspace paths

part1() {
  echo "############################################################"
  echo "# PART 1 — the POLICY ENGINE (broker_policy.py) on its own"
  echo "#   pure decision logic: human ∩ agent intersection"
  echo "############################################################"
  python broker_policy.py
}

part2() {
  echo "############################################################"
  echo "# PART 2 — the BROKER (broker_demo.py), the MCP tool"
  echo "#   policy decision -> scoped token OR structured denial"
  echo "############################################################"
  python -c "
import asyncio, broker_demo
async def go():
    for r,a,why in [('github','read','read issues'),
                    ('github','write','fix bug 42'),
                    ('notion','read','sync docs'),
                    ('stripe','read','billing peek')]:
        res = await broker_demo.request_credentials(r,a,reason=why)
        if res.granted:
            print(f'  GRANT {r}/{a:5} -> token={res.token}')
        else:
            print(f'  DENY  {r}/{a:5} -> {res.denied_reason}')
asyncio.run(go())
" 2>/dev/null
}

part3() {
  echo "############################################################"
  echo "# PART 3 — the FULL SANDBOX LOOP (agent_in_box.py in Docker)"
  echo "#   agent trapped in a box: direct read fails -> broker"
  echo "############################################################"
  if docker ps >/dev/null 2>&1; then
    docker run --rm -v "$(pwd)":/workspace -w /workspace python:3.12-slim \
      sh -c "pip install -q mcp pydantic pyyaml 2>/dev/null && python agent_in_box.py" 2>/dev/null
  else
    echo "  (skipped — Docker engine isn't running; launch Docker Desktop to see this part)"
  fi
}

part4() {
  echo "############################################################"
  echo "# PART 4 — the AUDIT LOG (audit.log) — persisted record"
  echo "#   every decision above, timestamped & leveled"
  echo "############################################################"
  cat audit.log
}

# Parse args: a leading --fresh wipes the audit log first; the rest are part numbers.
# By default the audit log is APPEND-ONLY (like a real audit trail) — never auto-deleted.
FRESH=0
PARTS=()
for arg in "$@"; do
  case "$arg" in
    --fresh) FRESH=1 ;;
    *)       PARTS+=("$arg") ;;
  esac
done
if [ ${#PARTS[@]} -eq 0 ]; then
  PARTS=(1 2 3 4)
fi

if [ "$FRESH" -eq 1 ]; then
  rm -f audit.log                   # explicit reset: start this run with a clean log
  echo "(--fresh: audit.log reset)"
fi

for p in "${PARTS[@]}"; do
  case "$p" in
    1) part1 ;;
    2) part2 ;;
    3) part3 ;;
    4) part4 ;;
    *) echo "unknown part: $p (valid: 1 2 3 4)" ;;
  esac
  echo ""
done
