import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .llm import LLMProvider


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


class SessionManager:
    def __init__(self, ttl_minutes: int = 60):
        self._sessions: dict[str, Session] = {}
        self._ttl = ttl_minutes * 60

    def get(self, phone: str) -> Session:
        from .llm import default_provider
        self._cleanup()
        if phone not in self._sessions:
            session = Session(phone=phone)
            session.llm_provider = default_provider()
            self._sessions[phone] = session
        session = self._sessions[phone]
        session.last_active = time.time()
        return session

    def reset(self, phone: str) -> None:
        self._sessions.pop(phone, None)

    def _cleanup(self) -> None:
        now = time.time()
        expired = [k for k, v in self._sessions.items() if now - v.last_active > self._ttl]
        for k in expired:
            del self._sessions[k]
