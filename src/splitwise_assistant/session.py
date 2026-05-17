import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from .llm import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ReceiptItem:
    name: str
    price: float
    payers: list[dict] = field(default_factory=list)  # [{"name": str, "user_id": int|None, "percentage": float}]


@dataclass
class Session:
    phone: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    history: list[dict] = field(default_factory=list)
    mode: str = "chat"  # "chat" | "receipt"
    receipt_items: list[ReceiptItem] = field(default_factory=list)
    current_item_index: int = 0
    group_id: Optional[int] = None
    friends: list[dict] = field(default_factory=list)  # cached from Splitwise
    # Lazy-initialised on first access via SessionManager
    llm_provider: Optional["LLMProvider"] = field(default=None, repr=False)
    # Per-user Splitwise credentials (not logged for security)
    splitwise_oauth_token: Optional[str] = field(default=None, repr=False)
    splitwise_api_key: Optional[str] = field(default=None, repr=False)
    # Per-user LLM API keys (not logged for security)
    anthropic_api_key: Optional[str] = field(default=None, repr=False)
    openai_api_key: Optional[str] = field(default=None, repr=False)
    # Per-session MCP bridge
    mcp_bridge: Optional[Any] = field(default=None, repr=False)
    # Whiteboard: cached group info to avoid repeated API calls
    whiteboard: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    def __init__(self, ttl_minutes: int = 60):
        self._sessions: dict[str, Session] = {}
        self._ttl = ttl_minutes * 60

    def get(self, phone: str) -> Session:
        # Note: _cleanup is now async, called separately in async contexts
        if phone not in self._sessions:
            session = Session(phone=phone)
            # Don't set llm_provider here - it will be set when credentials are provided
            self._sessions[phone] = session
        session = self._sessions[phone]
        session.last_active = time.time()
        return session

    async def set_credentials(
        self,
        phone: str,
        oauth_token: str | None = None,
        api_key: str | None = None,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None
    ) -> bool:
        """Set Splitwise and LLM credentials for a session.

        Args:
            phone: Session identifier
            oauth_token: Optional Splitwise OAuth access token
            api_key: Optional Splitwise API key
            anthropic_api_key: Optional Anthropic API key
            openai_api_key: Optional OpenAI API key

        Returns:
            True if credentials were validated and bridge initialized successfully

        Raises:
            ValueError: If no credentials provided or if credentials are invalid
        """
        # Validate at least one Splitwise credential and one LLM key provided
        has_splitwise = bool(oauth_token or api_key)
        has_llm = bool(anthropic_api_key or openai_api_key)

        if not has_splitwise:
            raise ValueError("Must provide Splitwise oauth_token or api_key")
        if not has_llm:
            raise ValueError("Must provide Anthropic or OpenAI API key")

        session = self.get(phone)

        # Clean up old bridge if exists
        if session.mcp_bridge:
            await session.mcp_bridge.shutdown()
            session.mcp_bridge = None

        # Store credentials (memory only)
        session.splitwise_oauth_token = oauth_token
        session.splitwise_api_key = api_key
        session.anthropic_api_key = anthropic_api_key
        session.openai_api_key = openai_api_key

        # Initialize LLM provider with session keys
        from .llm import make_provider_with_key
        if anthropic_api_key:
            session.llm_provider = make_provider_with_key("anthropic", "claude-sonnet-4-6", anthropic_api_key)
        elif openai_api_key:
            session.llm_provider = make_provider_with_key("openai", "gpt-4o", openai_api_key)

        # Create and initialize new bridge
        from .mcp_bridge import create_bridge
        bridge = create_bridge(
            oauth_access_token=oauth_token,
            api_key=api_key
        )

        try:
            await bridge.startup()
            session.mcp_bridge = bridge
            logger.info("Initialized bridge for session %s", phone)
            return True
        except Exception as e:
            logger.error("Failed to initialize bridge for %s: %s", phone, e)
            raise ValueError(f"Invalid credentials: {e}")

    async def reset(self, phone: str) -> None:
        """Reset session and cleanup resources."""
        session = self._sessions.get(phone)
        if session and session.mcp_bridge:
            await session.mcp_bridge.shutdown()
        self._sessions.pop(phone, None)

    async def _cleanup(self) -> None:
        """Cleanup expired sessions and their bridges."""
        now = time.time()
        expired = [
            k for k, v in self._sessions.items()
            if now - v.last_active > self._ttl
        ]
        for k in expired:
            session = self._sessions[k]
            if session.mcp_bridge:
                try:
                    await session.mcp_bridge.shutdown()
                except Exception as e:
                    logger.error("Error shutting down bridge for %s: %s", k, e)
            del self._sessions[k]
        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
