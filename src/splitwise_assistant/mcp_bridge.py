"""In-process FastMCP client bridge to the splitwise-mcp server."""

import logging
from typing import Any

from fastmcp import Client

from .config import settings

logger = logging.getLogger(__name__)


class MCPBridge:
    """Connects to the splitwise-mcp server in-process (no subprocess needed)."""

    def __init__(
        self,
        oauth_access_token: str | None = None,
        api_key: str | None = None
    ) -> None:
        self._oauth_access_token = oauth_access_token
        self._api_key = api_key
        self._client: Client | None = None
        self._tools_cache: list | None = None

    async def startup(self) -> None:
        """Initialize MCP client with instance-specific credentials.

        Credentials are passed directly to create_server() so each bridge gets
        its own SplitwiseClient with no shared global state between users.
        """
        from splitwise_mcp_server.server import create_server

        mcp_server = create_server(
            api_key=self._api_key,
            oauth_access_token=self._oauth_access_token,
        )

        # Client accepts a FastMCP instance directly — runs entirely in-process.
        self._client = Client(mcp_server)
        await self._client.__aenter__()
        self._tools_cache = await self._client.list_tools()
        logger.info("MCPBridge connected (in-process) — %d tools available", len(self._tools_cache))

    async def shutdown(self) -> None:
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None

    async def list_tools(self) -> list:
        if self._tools_cache is None:
            raise RuntimeError("MCPBridge not started")
        return self._tools_cache

    async def call_tool(self, name: str, arguments: dict) -> Any:
        if self._client is None:
            raise RuntimeError("MCPBridge not started")
        return await self._client.call_tool(name, arguments)

    def to_anthropic_tools(self) -> list[dict]:
        """Anthropic tool format with prompt caching on the last entry."""
        if self._tools_cache is None:
            return []
        tools = []
        for i, t in enumerate(self._tools_cache):
            schema = t.inputSchema if hasattr(t, "inputSchema") else {"type": "object", "properties": {}}
            tool: dict = {"name": t.name, "description": t.description or "", "input_schema": schema}
            if i == len(self._tools_cache) - 1:
                tool["cache_control"] = {"type": "ephemeral"}
            tools.append(tool)
        return tools

    def to_openai_tools(self, slim: bool = False) -> list[dict]:
        """OpenAI function-calling tool format.

        slim=True uses only core tools with truncated descriptions to fit
        low-TPM providers like Groq free tier.
        """
        if self._tools_cache is None:
            return []
        tools = self._tools_cache
        if slim:
            tools = [t for t in tools if t.name in _CORE_TOOL_NAMES]
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": _short_desc(t.description, slim),
                    "parameters": t.inputSchema if hasattr(t, "inputSchema") else {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]


# Tools included when slim=True (covers the vast majority of use cases)
_CORE_TOOL_NAMES = {
    "get_current_user", "get_groups", "get_group", "get_friends", "get_friend",
    "create_expense", "get_expenses", "get_expense", "update_expense", "delete_expense",
    "resolve_group", "resolve_friend",
}


def _short_desc(desc: str | None, slim: bool) -> str:
    if not desc:
        return ""
    if not slim:
        return desc
    # Keep only the first sentence to save tokens
    return desc.split(".")[0].strip()[:120]


def create_bridge(
    oauth_access_token: str | None = None,
    api_key: str | None = None
) -> MCPBridge:
    """Factory to create MCPBridge with specific credentials.

    Args:
        oauth_access_token: Optional Splitwise OAuth access token
        api_key: Optional Splitwise API key

    Returns:
        MCPBridge instance with the provided credentials
    """
    return MCPBridge(
        oauth_access_token=oauth_access_token,
        api_key=api_key
    )
