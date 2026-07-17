FROM python:3.12-slim
WORKDIR /app

# deps installed at BUILD time → runtime needs no internet (survives --internal)
RUN pip install --no-cache-dir mcp pydantic pyyaml

# copy ONLY the code — NOT the policy/audit/secret.
# the broker gets agent_policy.yaml via a read-only mount (3c);
# the agent gets nothing sensitive at all — that's the whole point
COPY *.py /app/

