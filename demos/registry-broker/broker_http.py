"""
the same "Server" over the "Streamable HTTP" transport
"""

from broker_demo import mcp  # same FastMCP app: request_credentials + policy + audit


if __name__ == "__main__":
    # serve the SAME broker over HTTP instead of stdio.
    # default bind 127.0.0.1:8000, endpoint path /mcp
    mcp.run(transport="streamable-http")
