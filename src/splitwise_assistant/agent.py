"""Claude/OpenAI agentic loop that uses splitwise-mcp tools."""

import json
import logging
from typing import Any

from .llm import LLMProvider
from .mcp_bridge import bridge
from .session import Session

logger = logging.getLogger(__name__)


async def run_agent(session: Session, user_message: str) -> str:
    """Run one conversation turn through the LLM + MCP agentic loop."""
    provider: LLMProvider = session.llm_provider

    session.history.append({"role": "user", "content": user_message})

    max_hist = getattr(provider, "max_history", 20)
    if len(session.history) > max_hist:
        session.history = session.history[-max_hist:]

    if provider.__class__.__name__ == "AnthropicProvider":
        tools = bridge.to_anthropic_tools()
    else:
        slim = getattr(provider, "_slim_tools", False)
        tools = bridge.to_openai_tools(slim=slim)

    for _ in range(10):  # safety cap on tool-call rounds
        try:
            response = provider.chat(session.history, tools)
        except Exception as exc:
            # If the LLM rejects our request (bad tool format, rate limit exhausted, etc.)
            # roll back the last user message so history stays clean, then surface the error.
            logger.exception("LLM call failed")
            session.history.pop()  # remove the user message we just appended
            err = str(exc)
            if "credit balance" in err.lower() or "billing" in err.lower():
                return (
                    "Your Anthropic API credit is exhausted.\n"
                    "Switch to a free model: /model llama  or  /model groq"
                )
            if "429" in err or "rate" in err.lower():
                return "Rate limit reached. Please wait a moment and try again, or switch models with /model."
            if "404" in err or "not found" in err.lower():
                return (
                    f"Model not available for your account: {session.llm_provider.name}\n"
                    "Try /model llama (free) or /model claude."
                )
            if "tool_use_failed" in err or "400" in err:
                session.history = []
                return "The AI had trouble processing that. I've reset the conversation — please try again."
            return f"Something went wrong: {err[:120]}"

        if not response.has_tool_calls:
            provider.append_assistant(session.history, response)
            return response.text or "Done."

        # Validate tool names before executing — small models sometimes hallucinate names
        valid_names = {t.name for t in (await bridge.list_tools())}
        bad = [tc.name for tc in response.tool_calls if tc.name not in valid_names]
        if bad:
            logger.warning("Model hallucinated tool name(s): %s — resetting history", bad)
            session.history = []
            return "The AI called a tool that doesn't exist. Conversation reset — please try again."

        # Tool use round
        provider.append_assistant(session.history, response)
        results = await _execute_tools(response.tool_calls)
        provider.append_tool_results(session.history, results)

    return "Sorry, I couldn't complete that request. Please try again."


_MAX_TOOL_RESULT_CHARS = 50000

async def _execute_tools(tool_calls) -> list[tuple[str, str, bool]]:
    results = []
    for tc in tool_calls:
        logger.info("Calling tool %s with %s", tc.name, tc.input)
        try:
            raw = await bridge.call_tool(tc.name, tc.input)
            content = _serialize(raw)
            is_error = False
        except Exception as exc:
            logger.exception("Tool %s failed", tc.name)
            content = f"Error: {exc}"
            is_error = True
        if len(content) > _MAX_TOOL_RESULT_CHARS:
            content = content[:_MAX_TOOL_RESULT_CHARS] + "\n…(truncated)"
        results.append((tc.id, content, is_error))
    return results


# Fields that add bulk but are useless to the LLM
_STRIP_KEYS = {
    "picture", "avatar", "photo", "image", "large", "medium", "small",
    "registration_status", "custom_picture", "notifications_read",
    "notifications_count", "default_currency", "locale", "date_format",
    "default_group_id", "first_name_with_is", "email",
    "created_at", "updated_at", "whiteboard", "invite_link",
    "has_single_default_split", "auto_simplify", "cover_photo",
    "reminder_frequency", "change_type", "transaction_confirmed",
    "transaction_method", "transaction_id", "payment_system_account_id",
    "cashier", "entry_method", "creation_method", "receipt", "repayments",
}


def _compress(obj: Any) -> Any:
    """Recursively strip noisy fields and long URLs from Splitwise responses."""
    if isinstance(obj, dict):
        return {
            k: _compress(v)
            for k, v in obj.items()
            if k not in _STRIP_KEYS and not (isinstance(v, str) and v.startswith("http") and len(v) > 60)
        }
    if isinstance(obj, list):
        return [_compress(i) for i in obj]
    return obj


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return json.dumps(_compress(parsed), default=str)
        except (json.JSONDecodeError, TypeError):
            return value
    try:
        return json.dumps(_compress(value), default=str)
    except Exception:
        return str(value)


async def parse_with_haiku(prompt: str) -> str:
    """Low-cost Haiku call for parsing tasks (always uses Anthropic)."""
    import anthropic as _anthropic
    from .config import settings
    client = _anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
