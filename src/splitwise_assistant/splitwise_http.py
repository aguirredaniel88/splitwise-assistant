"""Lightweight direct Splitwise REST client — no MCP involved."""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://secure.splitwise.com/api/v3.0"


class SplitwiseDirectClient:
    """Thin async wrapper around the Splitwise REST API.

    Used by manual-mode endpoints so they never touch the MCP bridge.
    """

    def __init__(self, api_key: str | None = None, oauth_token: str | None = None) -> None:
        if not (api_key or oauth_token):
            raise ValueError("api_key or oauth_token required")
        self._headers = {"Authorization": f"Bearer {api_key or oauth_token}"}
        self._http = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def _get(self, path: str, **params) -> Any:
        r = await self._http.get(f"{_BASE}/{path}", headers=self._headers, params=params)
        r.raise_for_status()
        return r.json()

    async def _post(self, path: str, body: dict) -> Any:
        r = await self._http.post(f"{_BASE}/{path}", headers=self._headers, json=body)
        r.raise_for_status()
        return r.json()

    async def get_current_user(self) -> dict:
        return await self._get("get_current_user")

    async def get_groups(self) -> dict:
        return await self._get("get_groups")

    async def get_group(self, group_id: int) -> dict:
        return await self._get(f"get_group/{group_id}")

    async def create_expense(self, data: dict) -> dict:
        return await self._post("create_expense", data)
