"""Twilio WhatsApp webhook handler."""

import logging

from fastapi import APIRouter, Form, Response

from .agent import run_agent
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

    # Reset command
    if body.lower() in ("/reset", "reset", "start over"):
        sessions.reset(phone)
        reply = "Conversation reset. How can I help you?"
        return _twiml_response(reply)

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


def _twiml_response(text: str) -> Response:
    # Twilio has a ~1600 char limit per message segment; truncate gracefully
    if len(text) > _MAX_WHATSAPP_CHARS:
        text = text[: _MAX_WHATSAPP_CHARS - 3] + "..."

    # Escape XML special chars
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    xml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{safe}</Message></Response>'
    return Response(content=xml, media_type="application/xml")
