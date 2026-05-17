"""Claude/OpenAI agentic loop that uses splitwise-mcp tools."""

import json
import logging
from typing import Any

from .llm import LLMProvider
from .session import Session

logger = logging.getLogger(__name__)


async def run_agent(session: Session, user_message: str) -> str:
    """Run one conversation turn through the LLM + MCP agentic loop."""
    # Check if LLM provider is set
    if not session.llm_provider:
        return "⚠️ Please set up your API credentials first."

    provider: LLMProvider = session.llm_provider

    # Get bridge from session
    bridge = session.mcp_bridge
    if not bridge:
        return "⚠️ Splitwise not configured. Please set up your credentials."

    # Inject whiteboard context if available (limited for Groq due to token constraints)
    whiteboard_context = ""
    if session.whiteboard:
        is_groq = "groq" in provider.name.lower() or getattr(provider, "_slim_tools", False)
        if is_groq:
            # For Groq: only include group names, not full details (save tokens)
            group_names = {gid: data.get("group_name", "Unknown") for gid, data in session.whiteboard.items()}
            whiteboard_context = f"\n\n**Cached groups:** {', '.join(group_names.values())}"
        else:
            whiteboard_context = f"\n\n**Your Whiteboard (cached group info):**\n```json\n{_serialize(session.whiteboard)}\n```"

    session.history.append({"role": "user", "content": user_message + whiteboard_context})

    max_hist = getattr(provider, "max_history", 20)
    if len(session.history) > max_hist:
        session.history = session.history[-max_hist:]

    if provider.__class__.__name__ == "AnthropicProvider":
        tools = bridge.to_anthropic_tools()
    else:
        slim = getattr(provider, "_slim_tools", False)
        tools = bridge.to_openai_tools(slim=slim)

        # For Groq: filter to only essential tools to save tokens
        if slim:
            essential_tools = {
                "create_expense",
                "get_groups",
                "get_group",
                "get_current_user",
                "get_friends"
            }
            tools = [t for t in tools if t.get("function", {}).get("name") in essential_tools]
            logger.info(f"Filtered to {len(tools)} essential tools for Groq")

    for _ in range(10):  # safety cap on tool-call rounds
        try:
            response = provider.chat(session.history, tools)
        except Exception as exc:
            # If the LLM rejects our request (bad tool format, rate limit exhausted, etc.)
            # roll back the last user message so history stays clean, then surface the error.
            logger.exception("LLM call failed")
            session.history.pop()  # remove the user message we just appended
            err = str(exc)

            # Check for quota/credit issues
            if "quota" in err.lower() or "insufficient_quota" in err.lower():
                provider_name = "Anthropic" if "anthropic" in session.llm_provider.name.lower() else "OpenAI"
                return (
                    f"⚠️ Your {provider_name} API credits are exhausted.\n\n"
                    f"Please add credits to your {provider_name} account or switch to a different model.\n"
                    f"Current model: {session.llm_provider.name}"
                )

            if "credit balance" in err.lower() or "billing" in err.lower():
                provider_name = "Anthropic" if "anthropic" in session.llm_provider.name.lower() else "OpenAI"
                return (
                    f"⚠️ {provider_name} API billing issue detected.\n\n"
                    f"Please check your {provider_name} account billing settings.\n"
                    f"Current model: {session.llm_provider.name}"
                )

            if "413" in err or ("tokens per minute" in err.lower() and "request too large" in err.lower()):
                # Groq token limit exceeded - clear history and suggest reset
                session.history = []
                return (
                    "⚠️ Request too large for Groq's free tier (12k token limit).\n\n"
                    "Groq works best for short conversations. Your history has been cleared.\n"
                    "Try asking your question again, or consider using Claude/GPT for longer chats."
                )

            if "429" in err or "rate" in err.lower():
                return (
                    "⚠️ Rate limit reached. Please wait a moment and try again.\n"
                    f"Current model: {session.llm_provider.name}"
                )

            if "404" in err or "not found" in err.lower():
                return (
                    f"⚠️ Model not available: {session.llm_provider.name}\n"
                    "Try switching to a different model."
                )

            if "tool_use_failed" in err or "400" in err:
                session.history = []
                return "⚠️ The AI had trouble processing that. I've reset the conversation — please try again."

            return f"⚠️ Error: {err[:200]}"

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
        results = await _execute_tools(response.tool_calls, bridge, session)
        provider.append_tool_results(session.history, results)

    return "Sorry, I couldn't complete that request. Please try again."


_MAX_TOOL_RESULT_CHARS = 50000

async def _execute_tools(tool_calls, bridge, session=None) -> list[tuple[str, str, bool]]:
    """Execute tools using provided bridge and auto-cache group data."""
    results = []
    for tc in tool_calls:
        logger.info("Calling tool %s with %s", tc.name, tc.input)
        try:
            raw = await bridge.call_tool(tc.name, tc.input)
            content = _serialize(raw)
            is_error = False

            # Auto-cache group data in whiteboard
            if session and not is_error:
                _cache_group_data(tc.name, tc.input, raw, session)

        except Exception as exc:
            logger.exception("Tool %s failed", tc.name)
            content = f"Error: {exc}"
            is_error = True
        if len(content) > _MAX_TOOL_RESULT_CHARS:
            content = content[:_MAX_TOOL_RESULT_CHARS] + "\n…(truncated)"
        results.append((tc.id, content, is_error))
    return results


def _cache_group_data(tool_name: str, tool_input: dict, raw_result: Any, session: Session) -> None:
    """Cache group information in the whiteboard for future use."""
    try:
        if tool_name in ("get_group", "create_group"):
            # Parse the result
            import json
            data = json.loads(raw_result) if isinstance(raw_result, str) else raw_result

            # Extract group data
            if isinstance(data, dict):
                group = data.get("group", data)
                if isinstance(group, dict) and "id" in group:
                    group_id = str(group["id"])
                    group_name = group.get("name", "Unknown")

                    # Extract members with their info
                    members = []
                    default_percentages = {}
                    for member in group.get("members", []):
                        user_id = member.get("user_id") or member.get("id")
                        if user_id:
                            first = member.get("first_name", "")
                            last = member.get("last_name", "")
                            name = f"{first} {last}".strip() or member.get("email", "Unknown")
                            members.append({
                                "user_id": user_id,
                                "name": name,
                                "email": member.get("email")
                            })

                            # Get default balance/percentage if available
                            balance = member.get("balance", [])
                            if balance and isinstance(balance, list):
                                for bal in balance:
                                    amount = bal.get("amount")
                                    if amount:
                                        default_percentages[str(user_id)] = float(amount)

                    # Cache in whiteboard
                    session.whiteboard[group_id] = {
                        "group_name": group_name,
                        "members": members,
                        "default_percentages": default_percentages,
                        "simplify_by_default": group.get("simplify_by_default", True)
                    }
                    logger.info("Cached group %s (%s) in whiteboard with %d members",
                               group_id, group_name, len(members))

        elif tool_name == "get_groups":
            # Cache all groups
            import json
            data = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
            groups_list = data.get("groups", []) if isinstance(data, dict) else []

            for group in groups_list:
                if isinstance(group, dict) and "id" in group:
                    group_id = str(group["id"])
                    group_name = group.get("name", "Unknown")

                    # Basic info only (no members detail unless get_group is called)
                    if group_id not in session.whiteboard:
                        session.whiteboard[group_id] = {
                            "group_name": group_name,
                            "members": [],  # Will be populated when get_group is called
                            "default_percentages": {},
                            "simplify_by_default": group.get("simplify_by_default", True)
                        }

    except Exception as e:
        logger.warning("Failed to cache group data: %s", e)


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


async def parse_with_haiku(prompt: str, api_key: str | None = None) -> str:
    """Low-cost Haiku call for parsing tasks (always uses Anthropic).

    Args:
        prompt: The parsing prompt
        api_key: Anthropic API key (required if not in settings)
    """
    import anthropic as _anthropic
    from .config import settings

    key = api_key or settings.anthropic_api_key
    if not key:
        raise ValueError("Anthropic API key required for receipt parsing")

    client = _anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
