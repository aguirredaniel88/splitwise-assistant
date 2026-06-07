"""Web client API — REST endpoints for the browser chat UI."""

import base64
import logging
import uuid

from fastapi import APIRouter, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import run_agent
from .llm import AVAILABLE_MODELS, make_provider, resolve_model
from .receipt import assign_receipt_items, handle_assignment_response, start_receipt_flow_b64
from .session import SessionManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
_sessions = SessionManager()


# ── Request models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ModelRequest(BaseModel):
    model: str
    session_id: str


class ReceiptAssignRequest(BaseModel):
    session_id: str
    assignments: list[dict]  # [{"item_index": int, "payer_ids": [int|null]}]


class CredentialsRequest(BaseModel):
    session_id: str
    oauth_token: str | None = None
    api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    groq_api_key: str | None = None


class WhiteboardRequest(BaseModel):
    session_id: str
    whiteboard: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/credentials")
async def set_credentials(req: CredentialsRequest):
    """Set Splitwise and LLM API credentials for a session."""
    try:
        logger.info("Setting credentials for session %s", req.session_id)
        await _sessions.set_credentials(
            f"web:{req.session_id}",
            oauth_token=req.oauth_token,
            api_key=req.api_key,
            anthropic_api_key=req.anthropic_api_key,
            openai_api_key=req.openai_api_key,
            groq_api_key=req.groq_api_key
        )
        session = _sessions.get(f"web:{req.session_id}")
        has_llm = bool(session.llm_provider)
        has_splitwise = bool(session.mcp_bridge)

        return {
            "ok": True,
            "message": "Credentials validated successfully",
            "chat_available": has_llm,
            "manual_available": has_splitwise
        }
    except ValueError as e:
        logger.warning("Credential validation failed: %s", e)
        return {"ok": False, "error": str(e)}
    except Exception as e:
        logger.exception("Error setting credentials for session %s", req.session_id)
        return {"ok": False, "error": f"Failed to validate credentials: {str(e)}"}


@router.get("/credentials/status")
async def credentials_status(session_id: str):
    """Check if credentials are configured and bridge is ready."""
    session = _sessions.get(f"web:{session_id}")
    has_splitwise = bool(session.splitwise_oauth_token or session.splitwise_api_key)
    has_llm = bool(session.llm_provider)
    has_bridge = session.mcp_bridge is not None

    tools_count = 0
    if has_bridge:
        try:
            tools_count = len(await session.mcp_bridge.list_tools())
        except:
            pass

    return {
        "configured": has_splitwise,  # Only Splitwise is required
        "ready": has_bridge,
        "tools_available": tools_count,
        "chat_available": has_llm,
        "manual_available": has_splitwise
    }


@router.get("/whiteboard")
async def get_whiteboard(session_id: str):
    """Get the whiteboard (cached group data) for a session."""
    session = _sessions.get(f"web:{session_id}")
    return {"whiteboard": session.whiteboard}


@router.post("/whiteboard")
async def set_whiteboard(req: WhiteboardRequest):
    """Set the whiteboard (restore from localStorage)."""
    session = _sessions.get(f"web:{req.session_id}")
    session.whiteboard = req.whiteboard


@router.get("/manual/current-user")
async def get_current_user_info(session_id: str):
    """Get current user information."""
    session = _sessions.get(f"web:{session_id}")
    bridge = session.mcp_bridge

    if not bridge:
        return {"ok": False, "error": "Splitwise not configured"}

    try:
        result = await bridge.call_tool("get_current_user", {})
        logger.info(f"get_current_user result type: {type(result)}")

        # Extract content from CallToolResult
        if hasattr(result, 'content'):
            raw_content = result.content
            logger.info(f"Has content attribute, type: {type(raw_content)}")
        else:
            raw_content = result
            logger.info(f"No content attribute, using result directly")

        # Parse content
        import json
        if isinstance(raw_content, str):
            logger.info(f"Raw content is string: {raw_content[:200]}")
            user = json.loads(raw_content)
        elif isinstance(raw_content, list) and len(raw_content) > 0:
            content_item = raw_content[0]
            logger.info(f"Raw content is list, first item type: {type(content_item)}")
            if hasattr(content_item, 'text'):
                logger.info(f"Content item has text: {content_item.text[:200]}")
                user = json.loads(content_item.text)
            else:
                user = json.loads(str(content_item))
        else:
            logger.info(f"Using raw_content as-is: {raw_content}")
            user = raw_content

        logger.info(f"Parsed user data: {user}")

        # Extract the nested user object if present
        if isinstance(user, dict) and "user" in user:
            user = user["user"]
            logger.info(f"Extracted nested user: {user}")

        # Clean up user data
        first_name = (user.get("first_name") or "").strip()
        last_name = (user.get("last_name") or "").strip()
        email = (user.get("email") or "").strip()

        result_user = {
            "id": user.get("id"),
            "first_name": first_name or None,
            "last_name": last_name or None,
            "email": email or None
        }
        logger.info(f"Returning user: {result_user}")

        return {
            "ok": True,
            "user": result_user
        }

    except Exception as e:
        logger.exception("Failed to get current user")
        return {"ok": False, "error": str(e)}


@router.get("/manual/groups")
async def load_groups(session_id: str):
    """Load groups from Splitwise and populate whiteboard."""
    session = _sessions.get(f"web:{session_id}")
    bridge = session.mcp_bridge

    if not bridge:
        return {"ok": False, "error": "Splitwise not configured", "groups": []}

    try:
        # Call get_groups tool
        result = await bridge.call_tool("get_groups", {})

        # Parse result - CallToolResult has content property
        import json

        # Extract content from CallToolResult object
        if hasattr(result, 'content'):
            raw_content = result.content
        elif hasattr(result, '__dict__'):
            raw_content = str(result)
        else:
            raw_content = result

        logger.info(f"get_groups raw result type: {type(result)}, content type: {type(raw_content)}")

        # Parse JSON
        if isinstance(raw_content, str):
            data = json.loads(raw_content)
        elif isinstance(raw_content, list) and len(raw_content) > 0:
            # CallToolResult.content might be a list with text content
            content_item = raw_content[0]
            if hasattr(content_item, 'text'):
                data = json.loads(content_item.text)
            else:
                data = json.loads(str(content_item))
        else:
            data = raw_content

        groups_list = data.get("groups", []) if isinstance(data, dict) else []
        logger.info(f"Found {len(groups_list)} groups in response")

        # Cache basic group info in whiteboard
        for group in groups_list:
            if isinstance(group, dict) and "id" in group:
                group_id = str(group["id"])

                # Get detailed group info to populate members
                try:
                    group_detail = await bridge.call_tool("get_group", {"group_id": int(group_id)})

                    # Extract content from CallToolResult
                    if hasattr(group_detail, 'content'):
                        raw_detail = group_detail.content
                    else:
                        raw_detail = group_detail

                    # Parse content
                    if isinstance(raw_detail, str):
                        detail_data = json.loads(raw_detail)
                    elif isinstance(raw_detail, list) and len(raw_detail) > 0:
                        content_item = raw_detail[0]
                        if hasattr(content_item, 'text'):
                            detail_data = json.loads(content_item.text)
                        else:
                            detail_data = json.loads(str(content_item))
                    else:
                        detail_data = raw_detail

                    group_info = detail_data.get("group", detail_data) if isinstance(detail_data, dict) else {}
                    logger.info(f"Group {group_id}: found {len(group_info.get('members', []))} members")

                    members = []
                    default_percentages = {}

                    for member in group_info.get("members", []):
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

                            # Get balance for default percentages
                            balance = member.get("balance", [])
                            if balance and isinstance(balance, list):
                                for bal in balance:
                                    amount = bal.get("amount")
                                    if amount:
                                        default_percentages[str(user_id)] = float(amount)

                    session.whiteboard[group_id] = {
                        "group_name": group_info.get("name", group.get("name", "Unknown")),
                        "members": members,
                        "default_percentages": default_percentages,
                        "simplify_by_default": group_info.get("simplify_by_default", True)
                    }
                except Exception as e:
                    logger.warning(f"Failed to load details for group {group_id}: {e}")
                    # Fall back to basic info
                    session.whiteboard[group_id] = {
                        "group_name": group.get("name", "Unknown"),
                        "members": [],
                        "default_percentages": {},
                        "simplify_by_default": group.get("simplify_by_default", True)
                    }

        logger.info(f"Loaded {len(session.whiteboard)} groups into whiteboard for session {session_id}")
        return {
            "ok": True,
            "whiteboard": session.whiteboard,
            "count": len(session.whiteboard)
        }

    except Exception as e:
        logger.exception("Failed to load groups")
        return {"ok": False, "error": str(e), "groups": []}


@router.post("/manual/expense")
async def create_manual_expense(data: dict):
    """Create Splitwise expense from manual UI.

    Expected payload:
    {
        "session_id": str,
        "group_id": int|None,
        "description": str,
        "cost": float,
        "currency_code": str,
        "split_method": "percentage" | "shares" | "fixed",
        "payers": [{"user_id": int|None, "paid": float}],
        "splits": [{"user_id": int|None, "value": float}]
    }
    """
    session_id = data.get("session_id", "")
    session = _sessions.get(f"web:{session_id}")
    bridge = session.mcp_bridge

    if not bridge:
        return {"ok": False, "error": "Splitwise not configured"}

    # Validate inputs
    description = data.get("description", "").strip()
    if not description:
        return {"ok": False, "error": "Description is required"}

    try:
        cost = float(data.get("cost", 0))
        if cost <= 0:
            return {"ok": False, "error": "Cost must be greater than 0"}
    except (TypeError, ValueError):
        return {"ok": False, "error": "Invalid cost value"}

    currency_code = data.get("currency_code", "USD")
    split_method = data.get("split_method", "percentage")
    payers = data.get("payers", [])
    splits = data.get("splits", [])

    if not payers:
        return {"ok": False, "error": "At least one payer is required"}

    if not splits:
        return {"ok": False, "error": "At least one split is required"}

    # Get current user ID
    import json
    try:
        me_result = await bridge.call_tool("get_current_user", {})

        # Extract content from CallToolResult
        if hasattr(me_result, 'content'):
            raw_me = me_result.content
        else:
            raw_me = me_result

        # Parse content
        if isinstance(raw_me, str):
            me = json.loads(raw_me)
        elif isinstance(raw_me, list) and len(raw_me) > 0:
            content_item = raw_me[0]
            if hasattr(content_item, 'text'):
                me = json.loads(content_item.text)
            else:
                me = json.loads(str(content_item))
        else:
            me = raw_me

        # Extract nested user object if present
        if isinstance(me, dict) and "user" in me:
            me = me["user"]

        current_user_id = me.get("id")
    except Exception as e:
        logger.exception("Failed to get current user")
        return {"ok": False, "error": f"Failed to get current user: {str(e)}"}

    # Build user list with paid_share and owed_share
    user_map = {}  # user_id -> {"paid_share": float, "owed_share": float}

    # Process payers (who paid)
    for payer in payers:
        paid_amount = float(payer.get("paid", 0))
        if paid_amount <= 0:
            continue
        user_id = payer.get("user_id") or current_user_id
        if user_id not in user_map:
            user_map[user_id] = {"paid_share": 0.0, "owed_share": 0.0}
        user_map[user_id]["paid_share"] = round(paid_amount, 2)

    # Process splits (who owes)
    total_owed_check = 0.0
    split_values = []  # For shares mode, we need to calculate proportions

    for split in splits:
        value = float(split.get("value", 0))
        if value <= 0:
            continue

        split_values.append({
            "user_id": split.get("user_id") or current_user_id,
            "value": value
        })
        total_owed_check += value

    # Calculate owed shares based on split method
    for split_data in split_values:
        user_id = split_data["user_id"]
        value = split_data["value"]

        if split_method == "percentage":
            owed_share = round(cost * value / 100, 2)
        elif split_method == "shares":
            # Proportional split: cost * (user_shares / total_shares)
            owed_share = round(cost * value / total_owed_check, 2)
        else:  # fixed amounts
            owed_share = round(value, 2)

        if user_id not in user_map:
            user_map[user_id] = {"paid_share": 0.0, "owed_share": 0.0}
        user_map[user_id]["owed_share"] = owed_share

    # Validate totals
    if split_method == "percentage":
        if abs(total_owed_check - 100) > 0.01:
            return {"ok": False, "error": f"Percentages must add up to 100 (currently {total_owed_check:.1f}%)"}
    elif split_method == "shares":
        # Shares mode is always valid as long as there are shares
        if total_owed_check <= 0:
            return {"ok": False, "error": "At least one share must be assigned"}
    else:  # fixed amounts
        if abs(total_owed_check - cost) > 0.01:
            return {"ok": False, "error": f"Amounts must add up to cost {cost:.2f} (currently {total_owed_check:.2f})"}

    # Build users list for Splitwise API
    users = []
    for user_id, shares in user_map.items():
        users.append({
            "user_id": user_id,
            "paid_share": str(shares["paid_share"]),
            "owed_share": str(shares["owed_share"]),
        })

    if not users:
        return {"ok": False, "error": "No valid users provided"}

    # Create expense via Splitwise
    try:
        args = {
            "description": description,
            "cost": str(round(cost, 2)),
            "currency_code": currency_code,
            "users": users,
        }

        group_id = data.get("group_id")
        if group_id:
            args["group_id"] = int(group_id)

        logger.info(f"Creating expense with args: {args}")
        result = await bridge.call_tool("create_expense", args)
        logger.info("Created manual expense: %s for $%s", description, cost)

        return {
            "ok": True,
            "message": f"✅ Expense '{description}' created successfully!",
            "expense": result
        }

    except Exception as e:
        logger.exception("Failed to create manual expense")
        return {"ok": False, "error": f"Failed to create expense: {str(e)}"}


@router.delete("/whiteboard")
async def clear_whiteboard(session_id: str):
    """Clear the whiteboard cache."""
    session = _sessions.get(f"web:{session_id}")
    session.whiteboard = {}
    return {"ok": True}


@router.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session = _sessions.get(f"web:{session_id}")
    body = req.message.strip()

    # Check credentials first
    if not session.mcp_bridge:
        return {
            "reply": "⚠️ Please set up your Splitwise credentials first.",
            "session_id": session_id,
            "needs_credentials": True
        }

    if body.lower() in ("reset", "/reset", "start over"):
        await _sessions.reset(f"web:{session_id}")
        return {"reply": "Conversation reset. How can I help you?", "session_id": session_id}

    if body.lower().startswith("/model"):
        reply = _model_command(body, session)
        return {"reply": reply, "session_id": session_id}

    try:
        if session.mode in ("receipt", "receipt_total") and body:
            reply = await handle_assignment_response(body, session)
        else:
            reply = await run_agent(session, body)
    except Exception as exc:
        logger.exception("Error in web chat for session %s", session_id)
        reply = f"Something went wrong: {str(exc)[:300]}"

    return {"reply": reply, "session_id": session_id}


@router.post("/chat/image")
async def chat_image(session_id: str = Form(...), file: UploadFile = File(...)):
    session = _sessions.get(f"web:{session_id}")
    receipt_data = None
    try:
        data = await file.read()
        b64 = base64.standard_b64encode(data).decode()
        media_type = (file.content_type or "image/jpeg").split(";")[0]
        reply = await start_receipt_flow_b64(b64, media_type, session)
        if session.mode == "receipt" and session.receipt_items:
            receipt_data = {
                "items": [{"name": item.name, "price": item.price} for item in session.receipt_items],
                "members": [{"id": f["id"], "name": f["name"]} for f in session.friends],
            }
    except Exception as exc:
        logger.exception("Error processing uploaded image")
        reply = f"Couldn't process the image: {str(exc)[:200]}"
    return {"reply": reply, "session_id": session_id, "receipt_data": receipt_data}


@router.post("/chat/receipt/assign")
async def receipt_assign(req: ReceiptAssignRequest):
    session = _sessions.get(f"web:{req.session_id}")
    if session.mode != "receipt" or not session.receipt_items:
        return {"reply": "No active receipt session.", "session_id": req.session_id}
    try:
        reply = await assign_receipt_items(session, req.assignments)
    except Exception as exc:
        logger.exception("Error creating receipt expenses")
        reply = f"Failed to create expenses: {str(exc)[:200]}"
    return {"reply": reply, "session_id": req.session_id}


@router.get("/chat/models")
async def get_models(session_id: str | None = None):
    current = "claude-sonnet-4-6"
    available = []

    if session_id:
        s = _sessions.get(f"web:{session_id}")
        if s.llm_provider:
            current = s.llm_provider.name

        # Only show models for which user has API keys
        for model_alias in AVAILABLE_MODELS:
            resolved = resolve_model(model_alias)
            if resolved:
                provider_name, _ = resolved
                if provider_name == "anthropic" and s.anthropic_api_key:
                    available.append(model_alias)
                elif provider_name == "openai" and s.openai_api_key:
                    available.append(model_alias)
                elif provider_name == "groq" and s.groq_api_key:
                    available.append(model_alias)
    else:
        available = sorted(AVAILABLE_MODELS)

    return {"current": current, "available": sorted(set(available))}


@router.post("/chat/model")
async def set_model(req: ModelRequest):
    session = _sessions.get(f"web:{req.session_id}")
    resolved = resolve_model(req.model)
    if not resolved:
        return {"ok": False, "error": f"Unknown model '{req.model}'"}
    provider_name, model_id = resolved

    # Use session API keys instead of global settings
    from .llm import make_provider_with_key
    if provider_name == "anthropic":
        if not session.anthropic_api_key:
            return {"ok": False, "error": "Anthropic API key not configured for this session"}
        session.llm_provider = make_provider_with_key(provider_name, model_id, session.anthropic_api_key)
    elif provider_name == "openai":
        if not session.openai_api_key:
            return {"ok": False, "error": "OpenAI API key not configured for this session"}
        session.llm_provider = make_provider_with_key(provider_name, model_id, session.openai_api_key)
    elif provider_name == "groq":
        if not session.groq_api_key:
            return {"ok": False, "error": "Groq API key not configured for this session"}
        session.llm_provider = make_provider_with_key(provider_name, model_id, session.groq_api_key)
    else:
        return {"ok": False, "error": f"Provider '{provider_name}' not supported"}

    session.history = []
    return {"ok": True, "model": model_id}


@router.delete("/chat")
async def reset_chat(session_id: str):
    await _sessions.reset(f"web:{session_id}")
    return {"ok": True}


def _model_command(body: str, session) -> str:
    from .llm import make_provider_with_key
    parts = body.strip().split(None, 1)
    current = session.llm_provider.name if session.llm_provider else "unknown"
    if len(parts) == 1:
        return f"Current model: {current}\n\nAvailable: {', '.join(sorted(AVAILABLE_MODELS))}"
    alias = parts[1].strip()
    resolved = resolve_model(alias)
    if not resolved:
        return f"Unknown model '{alias}'. Available: {', '.join(sorted(AVAILABLE_MODELS))}"
    provider_name, model_id = resolved

    # Use session API keys
    if provider_name == "anthropic":
        if not session.anthropic_api_key:
            return "⚠️ Anthropic API key not configured. Please reconnect with your keys."
        session.llm_provider = make_provider_with_key(provider_name, model_id, session.anthropic_api_key)
    elif provider_name == "openai":
        if not session.openai_api_key:
            return "⚠️ OpenAI API key not configured. Please reconnect with your keys."
        session.llm_provider = make_provider_with_key(provider_name, model_id, session.openai_api_key)
    elif provider_name == "groq":
        if not session.groq_api_key:
            return "⚠️ Groq API key not configured. Please reconnect with your keys."
        session.llm_provider = make_provider_with_key(provider_name, model_id, session.groq_api_key)
    else:
        return f"⚠️ Provider '{provider_name}' not supported"

    session.history = []
    return f"Switched to {model_id}. Conversation history cleared."


# ── UI ────────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def ui():
    return _HTML


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Splitwise Assistant</title>

<!-- PWA Meta Tags -->
<meta name="description" content="AI-powered Splitwise expense management assistant">
<meta name="theme-color" content="#5c6ef8">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Splitwise">
<link rel="manifest" href="/api/static/manifest.json">
<link rel="icon" type="image/svg+xml" href="/api/static/icon.svg">
<link rel="apple-touch-icon" href="/api/static/icon.svg">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #23263a;
    --border: #2e3250;
    --accent: #5c6ef8;
    --accent-h: #4a5be0;
    --text: #e8eaf6;
    --text2: #8b92b8;
    --user-bg: #5c6ef8;
    --bot-bg: #1e2235;
    --radius: 16px;
    --font: 'Segoe UI', system-ui, sans-serif;
  }

  body { background: var(--bg); color: var(--text); font-family: var(--font);
         display: flex; flex-direction: column; height: 100dvh; overflow: hidden; }

  header { display: flex; align-items: center; gap: 8px; padding: 12px 16px;
           background: var(--surface); border-bottom: 1px solid var(--border);
           flex-shrink: 0; flex-wrap: wrap; }
  header h1 { font-size: 1.1rem; font-weight: 600; flex: 1; min-width: 120px; }
  header h1 span { color: var(--accent); }

  select, button { font-family: var(--font); font-size: .85rem; cursor: pointer;
                   border: 1px solid var(--border); border-radius: 8px;
                   background: var(--surface2); color: var(--text); padding: 6px 12px;
                   transition: background .15s; white-space: nowrap; }
  select:hover, button:hover { background: var(--border); }

  #reset-btn { color: #f47; border-color: #f47; }
  #logout-btn { color: #f47; border-color: #f47; }

  .btn-text-short { display: none; }

  /* Mode toggle */
  .mode-toggle { display: flex; gap: 0; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }
  .mode-btn { padding: 6px 16px; background: var(--surface2); border: none; border-radius: 0;
              transition: background .15s, color .15s; }
  .mode-btn.active { background: var(--accent); color: white; }

  .hidden { display: none !important; }

  /* Mobile optimizations */
  @media (max-width: 600px) {
    header { gap: 6px; padding: 10px 12px; }
    header h1 { font-size: 1rem; }
    header h1 span { display: none; } /* Hide "Assistant" on mobile */
    select, button { font-size: .8rem; padding: 5px 8px; }
    #reset-btn, #logout-btn { padding: 5px 8px; }
    .btn-text-full { display: none; }
    .btn-text-short { display: inline; }
  }

  #chat-container { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

  #messages { flex: 1; overflow-y: auto; padding: 20px;
              display: flex; flex-direction: column; gap: 12px; }

  .msg { max-width: 78%; word-break: break-word; line-height: 1.55; }
  .msg.user { align-self: flex-end; background: var(--user-bg);
              color: #fff; border-radius: var(--radius) var(--radius) 4px var(--radius);
              padding: 10px 14px; }
  .msg.bot  { align-self: flex-start; background: var(--bot-bg);
              border: 1px solid var(--border);
              border-radius: var(--radius) var(--radius) var(--radius) 4px;
              padding: 10px 14px; white-space: pre-wrap; }
  .msg.bot.typing { color: var(--text2); font-style: italic; }
  .msg .img-preview { max-width: 220px; border-radius: 10px; margin-bottom: 6px; display: block; }

  /* ── Receipt assignment panel ─────────────────────────────────────────── */
  .receipt-panel { align-self: flex-start; max-width: min(520px, 92%);
                   background: var(--bot-bg); border: 1px solid var(--border);
                   border-radius: var(--radius) var(--radius) var(--radius) 4px;
                   padding: 14px 16px; }
  .receipt-panel-title { font-weight: 600; margin-bottom: 14px; font-size: .95rem; }
  .receipt-item { margin-bottom: 14px; }
  .receipt-item:last-of-type { margin-bottom: 0; }
  .item-label { font-size: .85rem; color: var(--text2); margin-bottom: 6px; display: flex;
                justify-content: space-between; }
  .item-label .item-name { color: var(--text); font-weight: 500; }
  .payer-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
  .payer-chip { padding: 6px 14px; border-radius: 20px; font-size: .85rem;
                border: 1px solid var(--border); background: var(--surface2);
                cursor: pointer; transition: background .15s, border-color .15s; color: var(--text); }
  .payer-chip:hover { background: var(--border); }
  .payer-chip.selected { background: var(--accent); border-color: var(--accent); color: #fff; }

  #payers-container, #splits-container { margin-top: 8px; }
  #payer-inputs-active { display: flex; flex-direction: column; gap: 10px; }
  #payer-validation { font-size: .85rem; padding: 8px; border-radius: 6px; text-align: center; font-weight: 500; }
  #payer-validation.valid { background: rgba(34, 170, 85, 0.2); border: 1px solid #2a5; color: #3fb; }
  #payer-validation.invalid { background: rgba(221, 68, 68, 0.2); border: 1px solid #d44; color: #f88; }
  #payer-validation.empty { display: none; }
  .receipt-divider { border: none; border-top: 1px solid var(--border); margin: 14px 0; }
  .receipt-submit { width: 100%; padding: 9px; background: var(--accent);
                    border-color: var(--accent); color: #fff; border-radius: 10px;
                    font-weight: 600; font-size: .9rem; margin-top: 14px; }
  .receipt-submit:hover:not(:disabled) { background: var(--accent-h); border-color: var(--accent-h); }
  .receipt-submit:disabled { opacity: .45; cursor: not-allowed; }

  /* ── Manual expense panel ──────────────────────────────────────────────── */
  #manual-panel {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    max-width: 600px;
    margin: 0 auto;
    width: 100%;
  }

  .manual-form {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .manual-form h2 {
    margin: 0 0 8px 0;
    font-size: 1.2rem;
    color: var(--accent);
  }

  .manual-form label {
    font-size: .9rem;
    color: var(--text2);
    margin-bottom: 4px;
    display: block;
  }

  .manual-form input, .manual-form select {
    padding: 10px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: var(--font);
    font-size: .9rem;
    width: 100%;
  }

  .manual-form input:focus, .manual-form select:focus {
    outline: none;
    border-color: var(--accent);
  }

  .split-method-toggle {
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 8px 0;
  }

  .split-method-toggle input[type="radio"] {
    width: auto;
    margin: 0;
  }

  .split-method-toggle label {
    margin: 0;
    color: var(--text);
    cursor: pointer;
  }

  #splits-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 8px;
  }

  .member-split-row {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .member-split-row label {
    flex: 1;
    font-size: .9rem;
    color: var(--text);
    margin: 0;
  }

  .member-split-row input {
    width: 100px;
    flex-shrink: 0;
  }

  #split-validation {
    font-size: .85rem;
    padding: 8px;
    border-radius: 6px;
    text-align: center;
    font-weight: 500;
  }

  #split-validation.valid {
    background: rgba(34, 170, 85, 0.2);
    border: 1px solid #2a5;
    color: #3fb;
  }

  #split-validation.invalid {
    background: rgba(221, 68, 68, 0.2);
    border: 1px solid #d44;
    color: #f88;
  }

  #split-validation.empty {
    display: none;
  }

  #create-expense-btn {
    width: 100%;
    padding: 12px;
    background: var(--accent);
    border: 1px solid var(--accent);
    color: #fff;
    border-radius: 10px;
    font-weight: 600;
    font-size: .95rem;
    margin-top: 4px;
    cursor: pointer;
    transition: background .15s;
  }

  #create-expense-btn:hover:not(:disabled) {
    background: var(--accent-h);
  }

  #create-expense-btn:disabled {
    opacity: .5;
    cursor: not-allowed;
  }

  /* ── Credentials modal ─────────────────────────────────────────────────── */
  .credentials-modal {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.85);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    overflow-y: auto;
    padding: 20px 0;
  }

  .modal-content {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    max-width: 500px;
    width: 90%;
    max-height: 90vh;
    overflow-y: auto;
    margin: auto;
  }

  .modal-content h2 {
    margin-bottom: 16px;
    color: var(--accent);
  }

  .modal-content p {
    margin-bottom: 12px;
    color: var(--text2);
    line-height: 1.5;
    font-size: .9rem;
  }

  .modal-content label {
    display: block;
    margin: 16px 0 6px;
    font-size: 0.9rem;
    font-weight: 500;
  }

  .modal-content input[type="text"],
  .modal-content input[type="password"] {
    width: 100%;
    padding: 10px 12px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: var(--font);
    font-size: 0.95rem;
  }

  .modal-content .divider {
    text-align: center;
    margin: 16px 0;
    color: var(--text2);
  }

  .modal-actions {
    display: flex;
    gap: 8px;
    margin-top: 20px;
  }

  .modal-actions button {
    flex: 1;
  }

  .error-msg {
    color: #f47;
    font-size: 0.85rem;
    margin-top: 8px;
  }

  /* Mobile modal optimizations */
  @media (max-width: 600px) {
    .credentials-modal {
      align-items: flex-start;
      padding: 10px 0;
    }
    .modal-content {
      width: 95%;
      padding: 20px 16px;
      max-height: 95vh;
      margin: 10px auto;
    }
    .modal-content h2 {
      font-size: 1.2rem;
    }
    .modal-content p {
      font-size: 0.85rem;
    }
    .modal-content label {
      font-size: 0.85rem;
      margin: 12px 0 4px;
    }
    .modal-content input[type="text"],
    .modal-content input[type="password"] {
      font-size: 0.9rem;
      padding: 8px 10px;
    }
    .modal-content .divider {
      margin: 12px 0;
      font-size: 0.85rem;
    }
    .modal-actions {
      flex-direction: column;
    }
    .modal-actions button {
      width: 100%;
    }
  }

  .input-row { display: flex; gap: 8px; padding: 14px 20px;
               background: var(--surface); border-top: 1px solid var(--border);
               flex-shrink: 0; align-items: flex-end; }
  #msg-input { flex: 1; background: var(--surface2); border: 1px solid var(--border);
               color: var(--text); border-radius: 12px; padding: 10px 14px;
               resize: none; max-height: 140px; font-size: .95rem; line-height: 1.4;
               font-family: var(--font); outline: none; }
  #msg-input:focus { border-color: var(--accent); }
  #send-btn { background: var(--accent); border-color: var(--accent); color: #fff;
              padding: 10px 18px; border-radius: 12px; font-weight: 600; }
  #send-btn:hover { background: var(--accent-h); border-color: var(--accent-h); }
  #img-btn, #mic-btn, #lang-btn { padding: 10px 12px; font-size: 1.1rem; border-radius: 12px; }
  #mic-btn.recording { background: #f47; border-color: #f47; color: #fff; animation: pulse 1.5s infinite; }
  #lang-btn { font-size: 0.85rem; padding: 8px 10px; }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }
  #file-input { display: none; }

  .drop-overlay { position: fixed; inset: 0; background: rgba(92,110,248,.15);
                  border: 3px dashed var(--accent); border-radius: 8px;
                  display: none; align-items: center; justify-content: center;
                  font-size: 1.4rem; color: var(--accent); z-index: 99; }
  body.dragging .drop-overlay { display: flex; }

  @media (max-width: 600px) { .msg { max-width: 92%; } }
</style>
</head>
<body>

<div class="drop-overlay">Drop receipt image to scan it</div>

<header>
  <h1>Splitwise <span>Assistant</span></h1>
  <div class="mode-toggle">
    <button class="mode-btn active" id="chat-mode-btn">Chat</button>
    <button class="mode-btn" id="manual-mode-btn">Manual</button>
  </div>
  <select id="model-select" title="Switch AI model"></select>
  <button id="install-btn" title="Install app" style="display:none"><span class="btn-text-full">Install App</span><span class="btn-text-short">📲</span></button>
  <button id="reset-btn" title="Start a new conversation"><span class="btn-text-full">New chat</span><span class="btn-text-short">New</span></button>
  <button id="logout-btn" title="Logout and clear all data"><span class="btn-text-full">Logout</span><span class="btn-text-short">🚪</span></button>
</header>

<div id="chat-container">
  <div id="messages"></div>

  <div class="input-row">
    <button id="img-btn" title="Upload a receipt">📷</button>
    <input type="file" id="file-input" accept="image/*">
    <button id="mic-btn" title="Voice input">🎤</button>
    <button id="lang-btn" title="Voice language">🇺🇸</button>
    <textarea id="msg-input" rows="1" placeholder="Ask about your expenses…"></textarea>
    <button id="send-btn">Send</button>
  </div>
</div>

<div id="manual-panel" class="hidden">
  <div class="manual-form">
    <h2>Create Expense</h2>

    <div>
      <label for="group-select">Group</label>
      <select id="group-select">
        <option value="">No group (personal)</option>
      </select>
    </div>

    <div>
      <label for="expense-desc">Description *</label>
      <input id="expense-desc" type="text" placeholder="e.g., Dinner at restaurant" required />
    </div>

    <div>
      <label for="expense-cost">Amount *</label>
      <input id="expense-cost" type="number" placeholder="0.00" step="0.01" min="0.01" required />
    </div>

    <div>
      <label for="expense-currency">Currency</label>
      <select id="expense-currency">
        <option value="COP">COP - Colombian Peso</option>
        <option value="USD">USD - US Dollar</option>
        <option value="MXN">MXN - Mexican Peso</option>
        <option value="EUR">EUR - Euro</option>
        <option value="GBP">GBP - British Pound</option>
        <option value="CAD">CAD - Canadian Dollar</option>
        <option value="AUD">AUD - Australian Dollar</option>
        <option value="BRL">BRL - Brazilian Real</option>
        <option value="ARS">ARS - Argentine Peso</option>
      </select>
    </div>

    <div>
      <label id="payers-label">Who paid?</label>
      <div id="payers-container"></div>
      <div id="payer-validation" class="empty"></div>
    </div>

    <div>
      <label>Split Method (who owes what)</label>
      <div class="split-method-toggle">
        <input type="radio" name="split-method" id="split-percentage" value="percentage" checked />
        <label for="split-percentage">Percentages (%)</label>
        <input type="radio" name="split-method" id="split-shares" value="shares" />
        <label for="split-shares">Shares</label>
        <input type="radio" name="split-method" id="split-fixed" value="fixed" />
        <label for="split-fixed">Fixed Amounts</label>
      </div>
    </div>

    <div>
      <label id="splits-label">Who owes what?</label>
      <div id="splits-container"></div>
    </div>

    <div id="split-validation" class="empty"></div>

    <button id="create-expense-btn" type="button">Create Expense</button>
  </div>
</div>

<script>
const API = '/api';
let sessionId = localStorage.getItem('sw_session') || null;

// ── Helpers ──────────────────────────────────────────────────────────────────
function saveSession(id) {
  sessionId = id;
  localStorage.setItem('sw_session', id);
}

function addMsg(role, text, imgSrc) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (imgSrc) {
    const img = document.createElement('img');
    img.src = imgSrc; img.className = 'img-preview';
    div.appendChild(img);
  }
  if (text) div.appendChild(document.createTextNode(text));
  document.getElementById('messages').appendChild(div);
  div.scrollIntoView({behavior: 'smooth', block: 'end'});
  return div;
}

function typing() {
  return addMsg('bot typing', 'Thinking…');
}

// ── Receipt assignment UI ────────────────────────────────────────────────────
function renderReceiptPanel(receiptData, sid) {
  const assignments = {};  // item_index -> Set of payer IDs (null = me, int = member)
  receiptData.items.forEach((_, i) => { assignments[i] = new Set(); });

  const panel = document.createElement('div');
  panel.className = 'receipt-panel';

  const title = document.createElement('div');
  title.className = 'receipt-panel-title';
  title.textContent = '🧾 Who pays for each item?';
  panel.appendChild(title);

  const submitBtn = document.createElement('button');
  submitBtn.className = 'receipt-submit';
  submitBtn.textContent = 'Create Expenses';
  submitBtn.disabled = true;

  function refreshSubmit() {
    const allDone = Object.values(assignments).every(s => s.size > 0);
    submitBtn.disabled = !allDone;
  }

  receiptData.items.forEach((item, idx) => {
    const itemDiv = document.createElement('div');
    itemDiv.className = 'receipt-item';

    const label = document.createElement('div');
    label.className = 'item-label';
    label.innerHTML = `<span class="item-name">${item.name}</span><span>$${item.price.toFixed(2)}</span>`;
    itemDiv.appendChild(label);

    const chips = document.createElement('div');
    chips.className = 'payer-chips';

    function makeChip(text, payerId) {
      const btn = document.createElement('button');
      btn.className = 'payer-chip';
      btn.textContent = text;
      btn.addEventListener('click', () => {
        if (assignments[idx].has(payerId)) {
          assignments[idx].delete(payerId);
          btn.classList.remove('selected');
        } else {
          assignments[idx].add(payerId);
          btn.classList.add('selected');
        }
        refreshSubmit();
      });
      return btn;
    }

    chips.appendChild(makeChip('Me', null));
    receiptData.members.forEach(m => chips.appendChild(makeChip(m.name, m.id)));

    itemDiv.appendChild(chips);
    panel.appendChild(itemDiv);

    if (idx < receiptData.items.length - 1) {
      const hr = document.createElement('hr');
      hr.className = 'receipt-divider';
      panel.appendChild(hr);
    }
  });

  submitBtn.addEventListener('click', async () => {
    submitBtn.disabled = true;
    submitBtn.textContent = 'Creating…';
    const assignmentList = Object.entries(assignments).map(([idx, payerSet]) => ({
      item_index: parseInt(idx),
      payer_ids: [...payerSet],
    }));
    try {
      const res = await fetch(`${API}/chat/receipt/assign`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id: sid, assignments: assignmentList}),
      });
      const data = await res.json();
      panel.remove();
      addMsg('bot', data.reply);
    } catch(err) {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Create Expenses';
      addMsg('bot', 'Failed to create expenses — please try again.');
    }
  });

  panel.appendChild(submitBtn);
  document.getElementById('messages').appendChild(panel);
  panel.scrollIntoView({behavior: 'smooth', block: 'end'});
}

// ── Models ───────────────────────────────────────────────────────────────────
async function loadModels() {
  const res = await fetch(`${API}/chat/models${sessionId ? '?session_id=' + sessionId : ''}`);
  const data = await res.json();
  const sel = document.getElementById('model-select');
  sel.innerHTML = '';
  data.available.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m; opt.textContent = m;
    if (m === data.current) opt.selected = true;
    sel.appendChild(opt);
  });
}

document.getElementById('model-select').addEventListener('change', async e => {
  if (!sessionId) return;
  const res = await fetch(`${API}/chat/model`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({model: e.target.value, session_id: sessionId})
  });
  const data = await res.json();
  if (data.ok) addMsg('bot', `Switched to ${data.model}. Conversation cleared.`);
  else addMsg('bot', data.error || 'Could not switch model.');
});

// ── Whiteboard (localStorage cache for groups) ──────────────────────────────
async function syncWhiteboard() {
  try {
    // Get server whiteboard
    const res = await fetch(`${API}/whiteboard?session_id=${sessionId}`);
    const data = await res.json();

    if (data.whiteboard && Object.keys(data.whiteboard).length > 0) {
      // Save to localStorage
      localStorage.setItem('sw_whiteboard', JSON.stringify(data.whiteboard));
    }
  } catch (err) {
    console.error('Failed to sync whiteboard:', err);
  }
}

async function restoreWhiteboard() {
  const stored = localStorage.getItem('sw_whiteboard');
  if (!stored) return;

  try {
    const whiteboard = JSON.parse(stored);
    await fetch(`${API}/whiteboard`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId, whiteboard})
    });
  } catch (err) {
    console.error('Failed to restore whiteboard:', err);
  }
}

// ── Send text ────────────────────────────────────────────────────────────────
async function sendText(text) {
  if (!text.trim()) return;
  addMsg('user', text);
  const t = typing();
  try {
    const res = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text, session_id: sessionId})
    });
    const data = await res.json();
    saveSession(data.session_id);
    t.remove();
    addMsg('bot', data.reply);

    // Sync whiteboard after each chat (might have cached new group data)
    await syncWhiteboard();
  } catch(err) {
    t.remove();
    addMsg('bot', 'Network error — please try again.');
  }
}

document.getElementById('send-btn').addEventListener('click', () => {
  const el = document.getElementById('msg-input');
  sendText(el.value);
  el.value = ''; el.style.height = '';
});

document.getElementById('msg-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    document.getElementById('send-btn').click();
  }
});

document.getElementById('msg-input').addEventListener('input', function() {
  this.style.height = '';
  this.style.height = Math.min(this.scrollHeight, 140) + 'px';
});

// ── Send image ───────────────────────────────────────────────────────────────
async function sendImage(file) {
  const url = URL.createObjectURL(file);
  addMsg('user', '', url);
  const t = typing();
  const fd = new FormData();
  fd.append('file', file);
  fd.append('session_id', sessionId || '');
  try {
    const res = await fetch(`${API}/chat/image`, {method: 'POST', body: fd});
    const data = await res.json();
    saveSession(data.session_id);
    t.remove();
    if (data.receipt_data) {
      renderReceiptPanel(data.receipt_data, data.session_id);
    } else {
      addMsg('bot', data.reply);
    }
  } catch(err) {
    t.remove();
    addMsg('bot', 'Could not process the image.');
  }
}

document.getElementById('img-btn').addEventListener('click', () =>
  document.getElementById('file-input').click());

document.getElementById('file-input').addEventListener('change', e => {
  if (e.target.files[0]) sendImage(e.target.files[0]);
  e.target.value = '';
});

// ── Drag & drop ──────────────────────────────────────────────────────────────
document.addEventListener('dragover', e => { e.preventDefault(); document.body.classList.add('dragging'); });
document.addEventListener('dragleave', e => { if (!e.relatedTarget) document.body.classList.remove('dragging'); });
document.addEventListener('drop', e => {
  e.preventDefault(); document.body.classList.remove('dragging');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) sendImage(file);
});

// ── Reset ────────────────────────────────────────────────────────────────────
document.getElementById('reset-btn').addEventListener('click', async () => {
  if (sessionId) await fetch(`${API}/chat?session_id=${sessionId}`, {method: 'DELETE'}).catch(()=>{});
  // Generate a fresh session ID so the server starts from a clean slate
  sessionId = crypto.randomUUID();
  localStorage.setItem('sw_session', sessionId);
  document.getElementById('messages').innerHTML = '';
  addMsg('bot', 'New conversation started. How can I help you?');
  await loadModels();
});

// ── Logout ───────────────────────────────────────────────────────────────────
document.getElementById('logout-btn').addEventListener('click', async () => {
  if (!confirm('Are you sure you want to logout? This will clear all your stored credentials and data.')) {
    return;
  }

  // Clear server-side session
  if (sessionId) {
    await fetch(`${API}/chat?session_id=${sessionId}`, {method: 'DELETE'}).catch(()=>{});
    await fetch(`${API}/whiteboard?session_id=${sessionId}`, {method: 'DELETE'}).catch(()=>{});
  }

  // Clear all localStorage data
  localStorage.removeItem('sw_session');
  localStorage.removeItem('sw_creds');
  localStorage.removeItem('sw_whiteboard');
  localStorage.removeItem('voice_lang');

  // Clear UI
  document.getElementById('messages').innerHTML = '';

  // Generate new session
  sessionId = crypto.randomUUID();
  localStorage.setItem('sw_session', sessionId);

  // Reset to defaults
  voiceLang = 'en-US';
  updateLangButton();
  document.getElementById('msg-input').placeholder = 'Ask about your expenses…';

  // Show credential modal again
  addMsg('bot', 'Logged out successfully. Please reconnect to continue.');
  showCredentialsModal();
});

// ── Credentials ──────────────────────────────────────────────────────────────
function showCredentialsModal() {
  const modal = document.createElement('div');
  modal.className = 'credentials-modal';
  modal.id = 'creds-modal';
  modal.innerHTML = `
    <div class="modal-content">
      <h2>Connect Your Accounts</h2>
      <p>Enter your API credentials to get started. Your keys are stored securely in your browser.</p>
      <p style="font-size: 0.85rem; color: var(--text2); margin-top: 8px;">
        <strong>Where to get keys:</strong><br>
        • Groq (FREE): <a href="https://console.groq.com/keys" target="_blank" style="color: var(--accent);">console.groq.com</a> ⭐<br>
        • Anthropic: <a href="https://console.anthropic.com/account/keys" target="_blank" style="color: var(--accent);">console.anthropic.com</a><br>
        • OpenAI: <a href="https://platform.openai.com/api-keys" target="_blank" style="color: var(--accent);">platform.openai.com</a><br>
        • Splitwise: <a href="https://secure.splitwise.com/apps" target="_blank" style="color: var(--accent);">secure.splitwise.com/apps</a>
      </p>

      <h3 style="margin-top: 20px; margin-bottom: 8px; font-size: 0.95rem;">Splitwise (required)</h3>
      <label for="api-key">Splitwise API Key (easiest)</label>
      <input type="password" id="api-key" placeholder="Get from secure.splitwise.com/apps">

      <div class="divider">OR</div>

      <label for="oauth-token">OAuth Access Token (advanced)</label>
      <input type="password" id="oauth-token" placeholder="Use OAuth setup script">

      <h3 style="margin-top: 24px; margin-bottom: 8px; font-size: 0.95rem;">LLM Provider (optional - for AI chat mode)</h3>
      <label for="groq-key">Groq API Key (FREE, recommended) ⭐</label>
      <input type="password" id="groq-key" placeholder="gsk_... (optional)">

      <div class="divider">OR</div>

      <label for="anthropic-key">Anthropic API Key</label>
      <input type="password" id="anthropic-key" placeholder="sk-ant-... (optional)">

      <div class="divider">OR</div>

      <label for="openai-key">OpenAI API Key</label>
      <input type="password" id="openai-key" placeholder="sk-... (optional)">

      <p style="font-size: 0.8rem; color: var(--text2); margin-top: 12px; font-style: italic;">
        💡 Skip LLM keys to use Manual mode only (no AI chat)
      </p>

      <h3 style="margin-top: 24px; margin-bottom: 8px; font-size: 0.95rem; display:none;">Splitwise (required)</h3>
      <label for="api-key">Splitwise API Key (easiest)</label>
      <input type="password" id="api-key" placeholder="Get from secure.splitwise.com/apps">

      <div class="divider">OR</div>

      <label for="oauth-token">OAuth Access Token (advanced)</label>
      <input type="password" id="oauth-token" placeholder="Use OAuth setup script">

      <div class="error-msg" id="cred-error" style="display:none"></div>

      <div class="modal-actions">
        <button id="save-creds" style="background:var(--accent); border-color:var(--accent); color:#fff">
          Connect
        </button>
        <button id="learn-more" onclick="window.open('https://secure.splitwise.com/apps', '_blank')">
          Get Keys
        </button>
      </div>
    </div>
  `;

  document.body.appendChild(modal);
  document.getElementById('save-creds').onclick = saveCredentials;
}

async function saveCredentials() {
  const groqKey = document.getElementById('groq-key').value.trim();
  const anthropicKey = document.getElementById('anthropic-key').value.trim();
  const openaiKey = document.getElementById('openai-key').value.trim();
  const oauthToken = document.getElementById('oauth-token').value.trim();
  const apiKey = document.getElementById('api-key').value.trim();
  const errorDiv = document.getElementById('cred-error');

  // Validate inputs
  const hasLLM = groqKey || anthropicKey || openaiKey;
  const hasSplitwise = oauthToken || apiKey;

  if (!hasSplitwise) {
    errorDiv.textContent = 'Please provide Splitwise OAuth token or API key';
    errorDiv.style.display = 'block';
    return;
  }

  const btn = document.getElementById('save-creds');
  btn.disabled = true;
  btn.textContent = 'Connecting...';

  try {
    const res = await fetch(`${API}/credentials`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session_id: sessionId,
        groq_api_key: groqKey || null,
        anthropic_api_key: anthropicKey || null,
        openai_api_key: openaiKey || null,
        oauth_token: oauthToken || null,
        api_key: apiKey || null
      })
    });

    const data = await res.json();

    if (data.ok) {
      // Store credentials in localStorage for persistence
      localStorage.setItem('sw_creds', JSON.stringify({
        groq_api_key: groqKey || null,
        anthropic_api_key: anthropicKey || null,
        openai_api_key: openaiKey || null,
        oauth_token: oauthToken || null,
        api_key: apiKey || null
      }));

      document.getElementById('creds-modal').remove();

      // Configure UI based on available modes
      const chatAvailable = data.chat_available;
      const manualAvailable = data.manual_available;

      if (chatAvailable && manualAvailable) {
        // Both modes available
        const provider = groqKey ? 'Groq (FREE)' : anthropicKey ? 'Anthropic' : 'OpenAI';
        addMsg('bot', `✅ Connected using ${provider}! How can I help you with your Splitwise expenses?`);
        document.getElementById('chat-mode-btn').style.display = '';
        document.getElementById('manual-mode-btn').style.display = '';
      } else if (manualAvailable) {
        // Manual mode only (no LLM)
        addMsg('bot', `✅ Connected to Splitwise! Switch to Manual mode to create expenses.`);
        document.getElementById('chat-mode-btn').style.display = 'none';
        document.getElementById('manual-mode-btn').style.display = '';
        // Auto-switch to manual mode
        document.getElementById('manual-mode-btn').click();
      }
    } else {
      errorDiv.textContent = data.error || 'Failed to connect';
      errorDiv.style.display = 'block';
      btn.disabled = false;
      btn.textContent = 'Connect';
    }
  } catch (err) {
    errorDiv.textContent = 'Network error - please try again';
    errorDiv.style.display = 'block';
    btn.disabled = false;
    btn.textContent = 'Connect';
  }
}

async function restoreCredentials() {
  const stored = localStorage.getItem('sw_creds');
  if (!stored) return false;

  try {
    const creds = JSON.parse(stored);
    const res = await fetch(`${API}/credentials`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session_id: sessionId,
        ...creds
      })
    });
    const data = await res.json();

    if (data.ok) {
      // Configure UI based on available modes
      if (!data.chat_available && data.manual_available) {
        // Manual mode only (no LLM)
        document.getElementById('chat-mode-btn').style.display = 'none';
        document.getElementById('manual-mode-btn').style.display = '';
      }
    }

    return data.ok;
  } catch {
    return false;
  }
}

// ── Voice Input ──────────────────────────────────────────────────────────────
let recognition = null;
let isRecording = false;
let voiceLang = 'en-US';  // Default language

// Auto-detect language from browser, fallback to saved preference
const browserLang = navigator.language || navigator.userLanguage;
if (browserLang.startsWith('es')) {
  voiceLang = 'es-ES';
} else if (localStorage.getItem('voice_lang')) {
  voiceLang = localStorage.getItem('voice_lang');
}

function updateLangButton() {
  const btn = document.getElementById('lang-btn');
  if (voiceLang === 'es-ES') {
    btn.textContent = '🇪🇸';
    btn.title = 'Voice: Spanish';
  } else {
    btn.textContent = '🇺🇸';
    btn.title = 'Voice: English';
  }
}

function initVoiceInput() {
  // Check for browser support
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.warn('Speech recognition not supported in this browser');
    document.getElementById('mic-btn').style.display = 'none';
    document.getElementById('lang-btn').style.display = 'none';
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = voiceLang;

  updateLangButton();

  recognition.onstart = () => {
    isRecording = true;
    document.getElementById('mic-btn').classList.add('recording');
    const listening = voiceLang === 'es-ES' ? 'Escuchando...' : 'Listening...';
    document.getElementById('msg-input').placeholder = listening;
  };

  recognition.onend = () => {
    isRecording = false;
    document.getElementById('mic-btn').classList.remove('recording');
    const placeholder = voiceLang === 'es-ES'
      ? 'Pregunta sobre tus gastos…'
      : 'Ask about your expenses…';
    document.getElementById('msg-input').placeholder = placeholder;
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    document.getElementById('msg-input').value = transcript;
    // Auto-send after voice input
    setTimeout(() => document.getElementById('send-btn').click(), 300);
  };

  recognition.onerror = (event) => {
    console.error('Speech recognition error:', event.error);
    isRecording = false;
    document.getElementById('mic-btn').classList.remove('recording');
    const placeholder = voiceLang === 'es-ES'
      ? 'Pregunta sobre tus gastos…'
      : 'Ask about your expenses…';
    document.getElementById('msg-input').placeholder = placeholder;

    if (event.error === 'not-allowed') {
      const msg = voiceLang === 'es-ES'
        ? '⚠️ Acceso al micrófono denegado. Por favor, permite el acceso al micrófono en la configuración del navegador.'
        : '⚠️ Microphone access denied. Please allow microphone access in your browser settings.';
      addMsg('bot', msg);
    }
  };
}

// Language toggle button
document.getElementById('lang-btn').addEventListener('click', () => {
  voiceLang = voiceLang === 'en-US' ? 'es-ES' : 'en-US';
  localStorage.setItem('voice_lang', voiceLang);
  if (recognition) {
    recognition.lang = voiceLang;
  }
  updateLangButton();

  // Update placeholder
  const placeholder = voiceLang === 'es-ES'
    ? 'Pregunta sobre tus gastos…'
    : 'Ask about your expenses…';
  document.getElementById('msg-input').placeholder = placeholder;
});

document.getElementById('mic-btn').addEventListener('click', () => {
  if (!recognition) {
    const msg = voiceLang === 'es-ES'
      ? '⚠️ Entrada de voz no compatible con este navegador. Prueba Chrome, Safari o Edge.'
      : '⚠️ Voice input not supported in this browser. Try Chrome, Safari, or Edge.';
    addMsg('bot', msg);
    return;
  }

  if (isRecording) {
    recognition.stop();
  } else {
    recognition.start();
  }
});

// ── Manual Expense Panel ────────────────────────────────────────────────────
let currentMode = 'chat';
let currentGroupData = null;
let currentUserInfo = null;  // Store current user info

// Mode toggle buttons
document.getElementById('chat-mode-btn').addEventListener('click', () => {
  currentMode = 'chat';
  document.getElementById('chat-container').classList.remove('hidden');
  document.getElementById('manual-panel').classList.add('hidden');
  document.getElementById('chat-mode-btn').classList.add('active');
  document.getElementById('manual-mode-btn').classList.remove('active');
});

document.getElementById('manual-mode-btn').addEventListener('click', async () => {
  currentMode = 'manual';
  document.getElementById('chat-container').classList.add('hidden');
  document.getElementById('manual-panel').classList.remove('hidden');
  document.getElementById('chat-mode-btn').classList.remove('active');
  document.getElementById('manual-mode-btn').classList.add('active');

  // Load groups if whiteboard is empty
  await loadGroupsIfNeeded();
  populateGroupSelect();

  // Initialize with "No group" selected - should only show current user
  const groupSelect = document.getElementById('group-select');
  if (groupSelect.value === '') {
    currentGroupData = null;
    renderPayerInputs([]);
    renderSplitInputs([]);
  }
});

// Load current user info
async function loadCurrentUser() {
  if (currentUserInfo) return; // Already loaded

  try {
    const res = await fetch(`${API}/manual/current-user?session_id=${sessionId}`);
    const data = await res.json();

    if (data.ok && data.user) {
      currentUserInfo = data.user;
      console.log('Current user:', currentUserInfo);
    }
  } catch (err) {
    console.error('Failed to load current user:', err);
  }
}

// Load groups from Splitwise if not already cached
async function loadGroupsIfNeeded() {
  // Always load current user info first (needed for payer selection)
  await loadCurrentUser();

  const stored = localStorage.getItem('sw_whiteboard');
  const whiteboard = stored ? JSON.parse(stored) : {};

  // If whiteboard is empty or has no groups, load from Splitwise
  if (Object.keys(whiteboard).length === 0) {
    const select = document.getElementById('group-select');
    select.disabled = true;
    select.innerHTML = '<option value="">Loading groups...</option>';

    try {
      const res = await fetch(`${API}/manual/groups?session_id=${sessionId}`);
      const data = await res.json();

      console.log('Load groups response:', data);

      if (data.ok && data.whiteboard) {
        // Save to localStorage
        localStorage.setItem('sw_whiteboard', JSON.stringify(data.whiteboard));
        console.log(`Loaded ${data.count} groups from Splitwise`, data.whiteboard);
      } else {
        console.warn('Failed to load groups:', data.error);
        alert('Failed to load groups: ' + (data.error || 'Unknown error'));
      }
    } catch (err) {
      console.error('Failed to load groups:', err);
      alert('Error loading groups: ' + err.message);
    } finally {
      select.disabled = false;
    }
  }
}

// Populate groups from whiteboard
function populateGroupSelect() {
  const stored = localStorage.getItem('sw_whiteboard');
  const whiteboard = stored ? JSON.parse(stored) : {};
  const select = document.getElementById('group-select');

  // Keep the "No group" option
  select.innerHTML = '<option value="">No group (personal)</option>';

  const groupCount = Object.keys(whiteboard).length;
  if (groupCount === 0) {
    const helperText = document.createElement('p');
    helperText.style.cssText = 'font-size:0.85rem; color:var(--text2); margin-top:8px; font-style:italic;';
    helperText.textContent = '💡 No groups found. You can still create personal expenses, or create a group at splitwise.com first.';
    select.parentElement.appendChild(helperText);
  } else {
    Object.entries(whiteboard).forEach(([groupId, data]) => {
      const option = document.createElement('option');
      option.value = groupId;
      option.textContent = data.group_name || `Group ${groupId}`;
      select.appendChild(option);
    });
  }
}

// Group selection change handler
document.getElementById('group-select').addEventListener('change', (e) => {
  const groupId = e.target.value;
  console.log('Group selected:', groupId);

  if (!groupId) {
    // Personal expense - only current user
    currentGroupData = null;
    renderPayerInputs([]);  // Empty array = only current user
    renderSplitInputs([]);
    return;
  }

  const stored = localStorage.getItem('sw_whiteboard');
  const whiteboard = stored ? JSON.parse(stored) : {};
  currentGroupData = whiteboard[groupId];

  console.log('Current group data:', currentGroupData);
  console.log('Members:', currentGroupData?.members);
  console.log('Default percentages:', currentGroupData?.default_percentages);

  if (currentGroupData && currentGroupData.members) {
    renderPayerInputs(currentGroupData.members);
    renderSplitInputs(currentGroupData.members);
    applyDefaultPercentages();
  } else {
    console.warn('No members found for group', groupId);
    renderPayerInputs([]);
    renderSplitInputs([]);
  }
});

// Render payer inputs (who paid) - button/chip style
function renderPayerInputs(members) {
  const container = document.getElementById('payers-container');
  container.innerHTML = '';

  // Get all members including current user
  const allMembers = [];

  // Add current user (match from members list if possible)
  if (currentUserInfo) {
    console.log('Current user ID:', currentUserInfo.id, 'Type:', typeof currentUserInfo.id);
    console.log('Members:', members.map(m => ({id: m.user_id, type: typeof m.user_id, name: m.name})));
    const matchedMember = members.find(m => String(m.user_id) === String(currentUserInfo.id));
    if (matchedMember) {
      // User is in the group, use their name from group
      allMembers.push({
        user_id: currentUserInfo.id,
        name: matchedMember.name,
        isCurrentUser: true
      });
    } else {
      // User not in group, use their account name
      const firstName = (currentUserInfo.first_name || '').trim();
      const lastName = (currentUserInfo.last_name || '').trim();
      let userName = `${firstName} ${lastName}`.trim();

      if (!userName && currentUserInfo.email) {
        userName = currentUserInfo.email.split('@')[0]; // Use email username
      }
      if (!userName) {
        userName = 'You';
      }
      allMembers.push({
        user_id: currentUserInfo.id,
        name: userName,
        isCurrentUser: true
      });
    }
  }

  // Add other members (excluding current user if already added)
  members.forEach(member => {
    if (!currentUserInfo || String(member.user_id) !== String(currentUserInfo.id)) {
      allMembers.push({
        user_id: member.user_id,
        name: member.name,
        isCurrentUser: false
      });
    }
  });

  // Create chips container
  const chipsDiv = document.createElement('div');
  chipsDiv.className = 'payer-chips';

  // Create input fields container
  const inputsDiv = document.createElement('div');
  inputsDiv.id = 'payer-inputs-active';

  // Track active payers
  const activePayers = new Set();

  // Default: current user pays full amount
  if (allMembers.length > 0 && allMembers[0].isCurrentUser) {
    activePayers.add(allMembers[0].user_id);
  }

  // Render chips
  allMembers.forEach(member => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'payer-chip';
    chip.dataset.userId = member.user_id;
    chip.textContent = member.name;

    if (activePayers.has(member.user_id)) {
      chip.classList.add('selected');
    }

    chip.addEventListener('click', () => {
      if (activePayers.has(member.user_id)) {
        activePayers.delete(member.user_id);
        chip.classList.remove('selected');
      } else {
        activePayers.add(member.user_id);
        chip.classList.add('selected');
      }
      updatePayerInputs();
    });

    chipsDiv.appendChild(chip);
  });

  container.appendChild(chipsDiv);
  container.appendChild(inputsDiv);

  // Function to update input fields based on selected payers
  function updatePayerInputs() {
    inputsDiv.innerHTML = '';
    const cost = parseFloat(document.getElementById('expense-cost').value) || 0;

    if (activePayers.size === 0) {
      validatePayers();
      return;
    }

    // If only one payer and cost > 0, default to full amount
    const defaultAmount = (activePayers.size === 1 && cost > 0) ? cost : 0;

    activePayers.forEach(userId => {
      const member = allMembers.find(m => m.user_id === userId);
      const row = document.createElement('div');
      row.className = 'member-split-row';
      row.innerHTML = `
        <label>${member.name}</label>
        <input type="number" class="payer-input" data-user-id="${userId}" step="0.01" min="0" value="${defaultAmount}" placeholder="0.00" />
      `;
      inputsDiv.appendChild(row);
    });

    // Add input listeners
    inputsDiv.querySelectorAll('.payer-input').forEach(input => {
      input.addEventListener('input', validatePayers);
    });

    validatePayers();
  }

  // Store for external access
  container._updatePayerInputs = updatePayerInputs;
  container._allMembers = allMembers;
  container._activePayers = activePayers;

  // Initial render
  updatePayerInputs();
}

// Validate payers add up to total cost
function validatePayers() {
  const cost = parseFloat(document.getElementById('expense-cost').value) || 0;
  const inputs = document.querySelectorAll('.payer-input');
  const validation = document.getElementById('payer-validation');

  let sum = 0;
  let hasNonZero = false;

  inputs.forEach(input => {
    const val = parseFloat(input.value) || 0;
    if (val > 0) hasNonZero = true;
    sum += val;
  });

  if (!hasNonZero) {
    validation.className = 'empty';
    validation.textContent = '';
    return;
  }

  if (cost <= 0) {
    validation.className = 'invalid';
    validation.textContent = '✗ Enter total amount first';
  } else if (Math.abs(sum - cost) <= 0.01) {
    validation.className = 'valid';
    validation.textContent = `✓ Payments total $${sum.toFixed(2)}`;
  } else {
    validation.className = 'invalid';
    validation.textContent = `✗ Payments must total $${cost.toFixed(2)} (currently $${sum.toFixed(2)})`;
  }
}

// Render split inputs for members (who owes)
function renderSplitInputs(members) {
  const container = document.getElementById('splits-container');
  container.innerHTML = '';

  // Get all members including current user (same logic as payers)
  const allMembers = [];

  // Add current user (match from members list if possible)
  if (currentUserInfo) {
    const matchedMember = members.find(m => String(m.user_id) === String(currentUserInfo.id));
    if (matchedMember) {
      // User is in the group, use their name from group
      allMembers.push({
        user_id: currentUserInfo.id,
        name: matchedMember.name,
        isCurrentUser: true
      });
    } else {
      // User not in group, use their account name
      const firstName = (currentUserInfo.first_name || '').trim();
      const lastName = (currentUserInfo.last_name || '').trim();
      let userName = `${firstName} ${lastName}`.trim();

      if (!userName && currentUserInfo.email) {
        userName = currentUserInfo.email.split('@')[0]; // Use email username
      }
      if (!userName) {
        userName = 'You';
      }
      allMembers.push({
        user_id: currentUserInfo.id,
        name: userName,
        isCurrentUser: true
      });
    }
  }

  // Add other members (excluding current user if already added)
  members.forEach(member => {
    if (!currentUserInfo || String(member.user_id) !== String(currentUserInfo.id)) {
      allMembers.push({
        user_id: member.user_id,
        name: member.name,
        isCurrentUser: false
      });
    }
  });

  // Render all members
  allMembers.forEach(member => {
    const row = document.createElement('div');
    row.className = 'member-split-row';
    row.innerHTML = `
      <label>${member.name}</label>
      <input type="number" class="split-input" data-user-id="${member.user_id}" step="0.01" min="0" value="0" />
    `;
    container.appendChild(row);
  });

  // Add input listeners for validation
  container.querySelectorAll('.split-input').forEach(input => {
    input.addEventListener('input', validateSplits);
  });

  validateSplits();
}

// Apply default percentages from whiteboard
function applyDefaultPercentages() {
  if (!currentGroupData || !currentGroupData.default_percentages) return;

  const defaultPercentages = currentGroupData.default_percentages;
  const total = Object.values(defaultPercentages).reduce((sum, val) => sum + Math.abs(val), 0);

  if (total === 0) return;

  // Normalize to percentages that add up to 100
  const inputs = document.querySelectorAll('.split-input');
  inputs.forEach(input => {
    const userId = input.dataset.userId;
    if (userId && defaultPercentages[userId]) {
      const rawAmount = Math.abs(defaultPercentages[userId]);
      const percentage = (rawAmount / total) * 100;
      input.value = percentage.toFixed(2);
    }
  });

  validateSplits();
}

// Split method toggle handler
document.querySelectorAll('input[name="split-method"]').forEach(radio => {
  radio.addEventListener('change', () => {
    updateSplitLabels();
    validateSplits();
  });
});

// Update split input labels based on method
function updateSplitLabels() {
  const method = document.querySelector('input[name="split-method"]:checked').value;
  const label = document.getElementById('splits-label');
  if (method === 'percentage') {
    label.textContent = 'Splits (%)';
  } else if (method === 'shares') {
    label.textContent = 'Splits (Shares)';
  } else {
    label.textContent = 'Splits (Fixed Amounts)';
  }
}

// Validate splits in real-time
function validateSplits() {
  const method = document.querySelector('input[name="split-method"]:checked').value;
  const cost = parseFloat(document.getElementById('expense-cost').value) || 0;
  const inputs = document.querySelectorAll('.split-input');
  const validation = document.getElementById('split-validation');

  let sum = 0;
  let hasNonZero = false;

  inputs.forEach(input => {
    const val = parseFloat(input.value) || 0;
    if (val > 0) hasNonZero = true;
    sum += val;
  });

  if (!hasNonZero) {
    validation.className = 'empty';
    validation.textContent = '';
    return;
  }

  if (method === 'percentage') {
    if (Math.abs(sum - 100) <= 0.01) {
      validation.className = 'valid';
      validation.textContent = `✓ Percentages total ${sum.toFixed(1)}%`;
    } else {
      validation.className = 'invalid';
      validation.textContent = `✗ Percentages must total 100% (currently ${sum.toFixed(1)}%)`;
    }
  } else if (method === 'shares') {
    // Shares mode - just show the total shares, always valid if > 0
    if (cost <= 0) {
      validation.className = 'invalid';
      validation.textContent = '✗ Enter an amount first';
    } else {
      validation.className = 'valid';
      validation.textContent = `✓ Total shares: ${sum.toFixed(1)} (each share = $${(cost / sum).toFixed(2)})`;
    }
  } else {
    // Fixed amounts mode
    if (cost <= 0) {
      validation.className = 'invalid';
      validation.textContent = '✗ Enter an amount first';
    } else if (Math.abs(sum - cost) <= 0.01) {
      validation.className = 'valid';
      validation.textContent = `✓ Amounts total $${sum.toFixed(2)}`;
    } else {
      validation.className = 'invalid';
      validation.textContent = `✗ Amounts must total $${cost.toFixed(2)} (currently $${sum.toFixed(2)})`;
    }
  }
}

// Validate when cost changes
document.getElementById('expense-cost').addEventListener('input', () => {
  validateSplits();

  // Update payer inputs if they exist
  const container = document.getElementById('payers-container');
  if (container && container._updatePayerInputs) {
    container._updatePayerInputs();
  } else {
    validatePayers();
  }
});

// Create expense button handler
document.getElementById('create-expense-btn').addEventListener('click', async () => {
  const description = document.getElementById('expense-desc').value.trim();
  const cost = parseFloat(document.getElementById('expense-cost').value) || 0;
  const currency = document.getElementById('expense-currency').value;
  const groupId = document.getElementById('group-select').value;
  const method = document.querySelector('input[name="split-method"]:checked').value;

  // Validation
  if (!description) {
    alert('Please enter a description');
    return;
  }

  if (cost <= 0) {
    alert('Please enter a valid amount');
    return;
  }

  // Collect payers (who paid)
  const payerInputs = document.querySelectorAll('.payer-input');
  const payers = [];

  payerInputs.forEach(input => {
    const value = parseFloat(input.value) || 0;
    if (value > 0) {
      payers.push({
        user_id: input.dataset.userId ? parseInt(input.dataset.userId) : null,
        paid: value
      });
    }
  });

  if (payers.length === 0) {
    alert('Please specify who paid for this expense');
    return;
  }

  // Validate payers add up to cost
  const payerTotal = payers.reduce((sum, p) => sum + p.paid, 0);
  if (Math.abs(payerTotal - cost) > 0.01) {
    alert(`Payments must add up to $${cost.toFixed(2)} (currently $${payerTotal.toFixed(2)})`);
    return;
  }

  // Collect splits (who owes)
  const splitInputs = document.querySelectorAll('.split-input');
  const splits = [];

  splitInputs.forEach(input => {
    const value = parseFloat(input.value) || 0;
    if (value > 0) {
      splits.push({
        user_id: input.dataset.userId ? parseInt(input.dataset.userId) : null,
        value: value
      });
    }
  });

  if (splits.length === 0) {
    alert('Please specify who owes what');
    return;
  }

  // Disable button during request
  const btn = document.getElementById('create-expense-btn');
  btn.disabled = true;
  btn.textContent = 'Creating...';

  try {
    const res = await fetch(`${API}/manual/expense`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session_id: sessionId,
        group_id: groupId || null,
        description: description,
        cost: cost,
        currency_code: currency,
        split_method: method,
        payers: payers,
        splits: splits
      })
    });

    const data = await res.json();

    if (data.ok) {
      // Success - show message and clear form
      alert(data.message || 'Expense created successfully!');
      document.getElementById('expense-desc').value = '';
      document.getElementById('expense-cost').value = '';
      renderPayerInputs(currentGroupData ? currentGroupData.members : []);
      renderSplitInputs(currentGroupData ? currentGroupData.members : []);
      if (currentGroupData) {
        applyDefaultPercentages();
      }
    } else {
      alert('Error: ' + (data.error || 'Failed to create expense'));
    }
  } catch (err) {
    console.error('Failed to create expense:', err);
    alert('Network error - please try again');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Create Expense';
  }
});

// ── Init ─────────────────────────────────────────────────────────────────────
(async function init() {
  // Ensure sessionId exists
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem('sw_session', sessionId);
  }

  loadModels();
  initVoiceInput();

  const restored = await restoreCredentials();
  if (!restored) {
    showCredentialsModal();
  } else {
    // Restore whiteboard cache from localStorage
    await restoreWhiteboard();
    addMsg('bot', 'Hi! Ask me anything about your Splitwise expenses, or drop a receipt photo.');
  }
})();

// ── PWA Installation ─────────────────────────────────────────────────────────
let deferredPrompt = null;

// Register service worker
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/api/static/sw.js')
    .then(reg => console.log('Service Worker registered'))
    .catch(err => console.log('Service Worker registration failed:', err));
}

// Capture the beforeinstallprompt event
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredPrompt = e;
  document.getElementById('install-btn').style.display = 'inline-block';
});

// Handle install button click
document.getElementById('install-btn').addEventListener('click', async () => {
  if (!deferredPrompt) return;

  deferredPrompt.prompt();
  const { outcome } = await deferredPrompt.userChoice;

  if (outcome === 'accepted') {
    console.log('User accepted the install prompt');
  }

  deferredPrompt = null;
  document.getElementById('install-btn').style.display = 'none';
});

// Hide install button if already installed
window.addEventListener('appinstalled', () => {
  console.log('PWA installed');
  document.getElementById('install-btn').style.display = 'none';
  deferredPrompt = null;
});

// For iOS - hide install button if running in standalone mode (already installed)
if (window.matchMedia('(display-mode: standalone)').matches ||
    window.navigator.standalone === true) {
  document.getElementById('install-btn').style.display = 'none';
}
</script>
</body>
</html>"""
