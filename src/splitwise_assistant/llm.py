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

SYSTEM_PROMPT = """You are a helpful Splitwise expense assistant on WhatsApp. You help users manage shared expenses with their friends and groups.

You have access to Splitwise tools. When the user asks about expenses, balances, or anything money-related, use the appropriate tools automatically.

Guidelines:
- Be concise — this is WhatsApp, keep replies short and clear.
- Format money as "$12.50" or "€10.00" depending on currency.
- When creating expenses, always confirm with the user first unless they give very clear instructions.
- If you need a friend or group name and it's ambiguous, use the resolve-friend or resolve-group tools.
- Use bullet points (•) instead of markdown headers for lists."""


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
    def __init__(self, model: str) -> None:
        self._model = model
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

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
    def __init__(self, model: str) -> None:
        self._model = model
        self._client = openai.OpenAI(api_key=settings.openai_api_key)

    @property
    def name(self) -> str:
        return self._model

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=1024,
            messages=full_messages,
            tools=tools or openai.NOT_GIVEN,
        )
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
    "claude": ("anthropic", "claude-sonnet-4-6"),
    "claude-sonnet": ("anthropic", "claude-sonnet-4-6"),
    "sonnet": ("anthropic", "claude-sonnet-4-6"),
    "claude-sonnet-4-6": ("anthropic", "claude-sonnet-4-6"),
    "claude-opus": ("anthropic", "claude-opus-4-7"),
    "opus": ("anthropic", "claude-opus-4-7"),
    "claude-haiku": ("anthropic", "claude-haiku-4-5-20251001"),
    "haiku": ("anthropic", "claude-haiku-4-5-20251001"),
    "gpt": ("openai", "gpt-4o"),
    "gpt-4o": ("openai", "gpt-4o"),
    "chatgpt": ("openai", "gpt-4o"),
    "openai": ("openai", "gpt-4o"),
    "gpt-mini": ("openai", "gpt-4o-mini"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
}

AVAILABLE_MODELS = sorted(_MODEL_ALIASES.keys())


def resolve_model(alias: str) -> tuple[str, str] | None:
    """Return (provider, model_id) for an alias, or None if unknown."""
    return _MODEL_ALIASES.get(alias.lower().strip())


def make_provider(provider: str, model: str) -> LLMProvider:
    if provider == "anthropic":
        return AnthropicProvider(model)
    if provider == "openai":
        return OpenAIProvider(model)
    raise ValueError(f"Unknown provider: {provider}")


def default_provider() -> LLMProvider:
    if settings.llm_provider == "openai":
        return OpenAIProvider(settings.openai_model)
    return AnthropicProvider(settings.anthropic_model)
