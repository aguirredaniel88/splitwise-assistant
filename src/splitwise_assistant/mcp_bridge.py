"""In-process FastMCP client bridge to the splitwise-mcp server."""

import logging
import os
from typing import Any

from fastmcp import Client

from .config import settings

logger = logging.getLogger(__name__)


class MCPBridge:
    """Connects to the splitwise-mcp server in-process (no subprocess needed)."""

    def __init__(self) -> None:
        self._client: Client | None = None
        self._tools_cache: list | None = None

    async def startup(self) -> None:
        # Inject Splitwise credentials into the process env before the MCP
        # server's lifespan reads them via SplitwiseConfig.from_env().
        if settings.splitwise_oauth_access_token:
            os.environ["SPLITWISE_OAUTH_ACCESS_TOKEN"] = settings.splitwise_oauth_access_token
        if settings.splitwise_api_key:
            os.environ["SPLITWISE_API_KEY"] = settings.splitwise_api_key

        from splitwise_mcp_server.server import create_server
        mcp_server = create_server()

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

    def to_openai_tools(self) -> list[dict]:
        """OpenAI function-calling tool format."""
        if self._tools_cache is None:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema if hasattr(t, "inputSchema") else {"type": "object", "properties": {}},
                },
            }
            for t in self._tools_cache
        ]


# Singleton used across the app
bridge = MCPBridge()
