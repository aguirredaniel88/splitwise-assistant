"""Twilio WhatsApp webhook handler."""

import logging

from fastapi import APIRouter, Form, Response

from .agent import run_agent
from .llm import AVAILABLE_MODELS, make_provider, resolve_model
from .receipt import handle_assignment_response, start_receipt_flow
from .session import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()
sessions = SessionManager()

_MAX_WHATSAPP_CHARS = 1500


@router.post("/webhook/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(""),
    NumMedia: int = Form(0),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None),
):
    phone = From
    session = sessions.get(phone)
    body = Body.strip()

    # Initialize session with global bridge if not already set
    if not session.mcp_bridge:
        from .main import _global_bridge
        session.mcp_bridge = _global_bridge

    # ── Built-in commands ────────────────────────────────────────────────────
    if body.lower() in ("/reset", "reset", "start over"):
        await sessions.reset(phone)
        return _twiml_response("Conversation reset. How can I help you?")

    if body.lower() in ("/help", "help", "ayuda", "commands"):
        return _twiml_response(
            "Available commands:\n"
            "• *reset* — start a fresh conversation\n"
            "• */model* — show current AI model\n"
            "• */model gemini* — switch to Gemini 2.0 Flash (free)\n"
            "• */model llama* — switch to Llama 3.3 via Groq (free)\n"
            "• */model claude* — switch to Claude Sonnet\n"
            "• */model gpt* — switch to GPT-4o\n\n"
            "Send a receipt photo to split it.\n"
            "Or just ask anything about your Splitwise expenses!"
        )

    if body.lower().startswith("/model"):
        return _twiml_response(_handle_model_command(body, session))

    # ── Normal routing ────────────────────────────────────────────────────────
    try:
        if NumMedia and NumMedia > 0 and MediaUrl0 and MediaContentType0 and "image" in MediaContentType0:
            reply = await start_receipt_flow(MediaUrl0, session)
        elif session.mode == "receipt" and body:
            reply = await handle_assignment_response(body, session)
        elif body:
            reply = await run_agent(session, body)
        else:
            reply = "Please send a message or a receipt photo."
    except Exception as exc:
        logger.exception("Error handling message from %s", phone)
        reply = f"Something went wrong: {str(exc)[:120]}\n\nSend 'reset' to start over."

    return _twiml_response(reply)


def _handle_model_command(body: str, session) -> str:
    parts = body.strip().split(None, 1)
    current = session.llm_provider.name if session.llm_provider else "unknown"

    if len(parts) == 1:
        # "/model" with no argument — show status
        aliases = ", ".join(AVAILABLE_MODELS)
        return f"Current model: {current}\n\nAvailable: {aliases}"

    alias = parts[1].strip()
    resolved = resolve_model(alias)
    if not resolved:
        return f"Unknown model '{alias}'.\n\nAvailable: {', '.join(AVAILABLE_MODELS)}"

    provider_name, model_id = resolved
    session.llm_provider = make_provider(provider_name, model_id)
    # Clear history — message formats are provider-specific
    session.history = []
    return f"Switched to {model_id}. Conversation history cleared."


def _twiml_response(text: str) -> Response:
    # Twilio has a ~1600 char limit per message segment; truncate gracefully
    if len(text) > _MAX_WHATSAPP_CHARS:
        text = text[: _MAX_WHATSAPP_CHARS - 3] + "..."

    # Escape XML special chars
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    xml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{safe}</Message></Response>'
    return Response(content=xml, media_type="application/xml")
