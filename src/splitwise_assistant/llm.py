"""LLM provider abstraction for Anthropic and OpenAI."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import anthropic
import openai

from .config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful Splitwise expense assistant. You help users manage shared expenses with their friends and groups.

You have access to Splitwise tools. When the user asks about expenses, balances, or anything money-related, use the appropriate tools automatically.

**IMPORTANT - Whiteboard Cache:**
- You have a "whiteboard" that stores recently used group information (members, default split percentages)
- ALWAYS check the whiteboard FIRST before calling get_group for groups you've used recently
- The whiteboard contains: group_id, group_name, members (with user_ids), and default_percentages
- When creating expenses in a cached group, use the whiteboard data directly - no need to call get_group again
- If the user mentions a group that's NOT in the whiteboard, then call get_group and the info will be cached automatically

**Expense Creation Guidelines:**
- Be concise — keep replies short and clear.
- Format money as "$12.50" or "€10.00" depending on currency.
- When creating group expenses:
  1. Check whiteboard for group info first
  2. If not cached, call get_group to get member list and default percentages
  3. Use default percentages from the group if available
  4. If no default percentages and user didn't specify split, ASK: "How should we split this? Equally or custom percentages?"
- Never ask for emails if people are already in the group — use their user_ids from the whiteboard or get_group result.
- If you need a friend or group name and it's ambiguous, use the resolve-friend or resolve-group tools.
- Use bullet points (•) instead of markdown headers for lists.
- Users can send /reset to start a fresh conversation."""


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(ABC):
    """Abstract LLM provider. Each implementation owns its message format."""

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        ...

    @abstractmethod
    def append_assistant(self, messages: list[dict], response: LLMResponse) -> None:
        """Append the assistant turn (with any tool calls) to messages in-place."""
        ...

    @abstractmethod
    def append_tool_results(self, messages: list[dict], results: list[tuple[str, str, bool]]) -> None:
        """Append tool results to messages in-place.
        results: list of (tool_call_id, content, is_error)
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key or settings.anthropic_api_key)

    @property
    def name(self) -> str:
        return self._model

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            tools=tools,
        )
        text = "\n".join(b.text for b in response.content if hasattr(b, "text")).strip()
        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=b.input)
            for b in response.content
            if b.type == "tool_use"
        ]
        return LLMResponse(text=text, tool_calls=tool_calls)

    def append_assistant(self, messages: list[dict], response: LLMResponse) -> None:
        content: list = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        for tc in response.tool_calls:
            content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
        messages.append({"role": "assistant", "content": content})

    def append_tool_results(self, messages: list[dict], results: list[tuple[str, str, bool]]) -> None:
        content = []
        for tool_call_id, result_content, is_error in results:
            entry: dict = {"type": "tool_result", "tool_use_id": tool_call_id, "content": result_content}
            if is_error:
                entry["is_error"] = True
            content.append(entry)
        messages.append({"role": "user", "content": content})


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        slim_tools: bool = False,
        max_history: int = 20,
    ) -> None:
        self._model = model
        self._slim_tools = slim_tools
        self.max_history = max_history
        self._client = openai.OpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url,
        )

    @property
    def name(self) -> str:
        return self._model

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        kwargs: dict = dict(
            model=self._model,
            max_tokens=1024,
            messages=full_messages,
            tools=tools or openai.NOT_GIVEN,
        )
        if self._slim_tools and tools:
            # Smaller models hallucinate less with parallel tool calls disabled
            kwargs["parallel_tool_calls"] = False
        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        text = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                ))
        return LLMResponse(text=text, tool_calls=tool_calls)

    def append_assistant(self, messages: list[dict], response: LLMResponse) -> None:
        msg: dict = {"role": "assistant", "content": response.text or None}
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.input)},
                }
                for tc in response.tool_calls
            ]
        messages.append(msg)

    def append_tool_results(self, messages: list[dict], results: list[tuple[str, str, bool]]) -> None:
        # OpenAI needs one message per tool result
        for tool_call_id, result_content, _ in results:
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result_content})


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Friendly name → (provider, model)
_MODEL_ALIASES: dict[str, tuple[str, str]] = {
    # Anthropic
    "claude": ("anthropic", "claude-sonnet-4-6"),
    "claude-sonnet": ("anthropic", "claude-sonnet-4-6"),
    "sonnet": ("anthropic", "claude-sonnet-4-6"),
    "claude-sonnet-4-6": ("anthropic", "claude-sonnet-4-6"),
    "claude-opus": ("anthropic", "claude-opus-4-7"),
    "opus": ("anthropic", "claude-opus-4-7"),
    "claude-haiku": ("anthropic", "claude-haiku-4-5-20251001"),
    "haiku": ("anthropic", "claude-haiku-4-5-20251001"),
    # OpenAI
    "gpt": ("openai", "gpt-4o"),
    "gpt-4o": ("openai", "gpt-4o"),
    "chatgpt": ("openai", "gpt-4o"),
    "openai": ("openai", "gpt-4o"),
    "gpt-mini": ("openai", "gpt-4o-mini"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    # Groq / Llama (FREE) - use small 8B model to fit within 12k token limit
    "groq": ("groq", "llama-3.1-8b-instant"),
    "llama": ("groq", "llama-3.1-8b-instant"),
    "llama3": ("groq", "llama-3.1-8b-instant"),
    "llama-3.1-8b": ("groq", "llama-3.1-8b-instant"),
    "llama-large": ("groq", "llama-3.3-70b-versatile"),
    "llama-3.3-70b": ("groq", "llama-3.3-70b-versatile"),
}

AVAILABLE_MODELS = sorted(_MODEL_ALIASES.keys())


def resolve_model(alias: str) -> tuple[str, str] | None:
    """Return (provider, model_id) for an alias, or None if unknown."""
    return _MODEL_ALIASES.get(alias.lower().strip())


def make_provider(provider: str, model: str) -> LLMProvider:
    """Create provider using global settings (for backward compatibility)."""
    if provider == "anthropic":
        return AnthropicProvider(model)
    if provider == "openai":
        return OpenAIProvider(model)
    raise ValueError(f"Unknown provider: {provider}")


def make_provider_with_key(provider: str, model: str, api_key: str) -> LLMProvider:
    """Create provider with session-specific API key."""
    if provider == "anthropic":
        return AnthropicProvider(model, api_key=api_key)
    if provider == "openai":
        return OpenAIProvider(model, api_key=api_key)
    if provider == "groq":
        return OpenAIProvider(
            model=model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            slim_tools=True,
            max_history=2,  # Groq has strict 12k token limit - keep history extremely short
        )
    raise ValueError(f"Unknown provider: {provider}")


def default_provider() -> LLMProvider:
    """Create default provider (used for WhatsApp, requires env vars)."""
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return OpenAIProvider(settings.openai_model)
    if settings.anthropic_api_key:
        return AnthropicProvider(settings.anthropic_model)
    # If no keys in settings, return a dummy provider (web UI will override with session keys)
    return AnthropicProvider(settings.anthropic_model)
