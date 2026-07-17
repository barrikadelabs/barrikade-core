"""
the same "Server" over the "Streamable HTTP" transport
"""

from broker_demo import mcp
from mcp.server.transport_security import TransportSecuritySettings


if __name__ == "__main__":
    mcp.settings.host = "0.0.0.0"
    # DNS-rebinding protection stays ON; we just allowlist the cohostname
    # the agent uses ("broker:8000"). localhost stays allowed so still works.
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["broker:8000", "broker:*", "127.0.0.1:*", "localhost:*"],
        allowed_origins=["http://broker:8000", "http://127.0.0.1:*", "http://localhost:*"],
    )
    mcp.run(transport="streamable-http")
