from mcp.server.fastmcp import FastMCP
from datetime import datetime, timezone

mcp = FastMCP("demo")          # names your server; the client sees "demo"

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""     # this docstring becomes the tool's description
    return a + b

@mcp.tool()
def get_time() -> str:
    """Return the current UTC time as an I"""
    return datetime.now(timezone.utc).isoformat()

if __name__ == "__main__":
    mcp.run()                  # starts the server on stdio transport