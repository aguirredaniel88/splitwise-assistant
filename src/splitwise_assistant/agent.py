"""Claude agentic loop that uses splitwise-mcp tools."""

import json
import logging
from typing import Any

import anthropic

from .config import settings
from .mcp_bridge import bridge
from .session import Session

logger = logging.getLogger(__name__)

_anthropic = anthropic.Anthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = """You are a helpful Splitwise expense assistant on WhatsApp. You help users manage shared expenses with their friends and groups.

You have access to Splitwise tools. When the user asks about expenses, balances, or anything money-related, use the appropriate tools automatically.

Guidelines:
- Be concise — this is WhatsApp, keep replies short and clear.
- Format money as "$12.50" or "€10.00" depending on currency.
- When creating expenses, always confirm with the user first unless they give very clear instructions.
- If you need a friend or group name and it's ambiguous, use the resolve-friend or resolve-group tools.
- Use bullet points (•) instead of markdown headers for lists."""


async def run_agent(session: Session, user_message: str) -> str:
    """Run one conversation turn through the Claude + MCP agentic loop."""
    session.history.append({"role": "user", "content": user_message})

    # Trim history to last 30 messages to avoid context overflow
    if len(session.history) > 30:
        session.history = session.history[-30:]

    tools = bridge.to_anthropic_tools()

    for _ in range(10):  # safety limit on tool-call iterations
        response = _anthropic.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=session.history,
            tools=tools,
        )

        if response.stop_reason == "end_turn":
            text = _extract_text(response.content)
            session.history.append({"role": "assistant", "content": response.content})
            return text

        if response.stop_reason == "tool_use":
            session.history.append({"role": "assistant", "content": response.content})
            tool_results = await _execute_tools(response.content)
            session.history.append({"role": "user", "content": tool_results})
            continue

        break

    return "Sorry, I couldn't complete that request. Please try again."


async def _execute_tools(content: list) -> list[dict]:
    results = []
    for block in content:
        if block.type != "tool_use":
            continue
        logger.info("Calling tool %s with %s", block.name, block.input)
        try:
            raw = await bridge.call_tool(block.name, block.input)
            result_text = _serialize(raw)
        except Exception as exc:
            logger.exception("Tool %s failed", block.name)
            result_text = f"Error: {exc}"

        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result_text,
        })
    return results


def _extract_text(content: list) -> str:
    return "\n".join(b.text for b in content if hasattr(b, "text")).strip()


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, indent=2)
    except Exception:
        return str(value)


async def parse_with_haiku(prompt: str) -> str:
    """Low-cost parsing helper using Haiku."""
    response = _anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
