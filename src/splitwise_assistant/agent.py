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

    # Keep last 30 messages to avoid context overflow
    if len(session.history) > 30:
        session.history = session.history[-30:]

    tools = (
        bridge.to_anthropic_tools()
        if provider.__class__.__name__ == "AnthropicProvider"
        else bridge.to_openai_tools()
    )

    for _ in range(10):  # safety cap on tool-call rounds
        response = provider.chat(session.history, tools)

        if not response.has_tool_calls:
            provider.append_assistant(session.history, response)
            return response.text or "Done."

        # Tool use round
        provider.append_assistant(session.history, response)
        results = await _execute_tools(response.tool_calls)
        provider.append_tool_results(session.history, results)

    return "Sorry, I couldn't complete that request. Please try again."


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
        results.append((tc.id, content, is_error))
    return results


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, indent=2)
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
