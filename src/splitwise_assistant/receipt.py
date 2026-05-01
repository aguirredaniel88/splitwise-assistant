"""Receipt image processing and interactive expense assignment flow."""

import base64
import json
import logging
from typing import Optional

import anthropic
import httpx

from .agent import parse_with_haiku
from .config import settings
from .mcp_bridge import bridge
from .session import Session, ReceiptItem

logger = logging.getLogger(__name__)

_anthropic = anthropic.Anthropic(api_key=settings.anthropic_api_key)

# Keywords that trigger "split total equally" shortcut
_TOTAL_KEYWORDS = {"total", "equally", "equal", "todo", "igual", "all", "together", "split total"}


async def start_receipt_flow_b64(image_b64: str, media_type: str, session: Session) -> str:
    """Start receipt flow from already-encoded base64 image (web client)."""
    return await _start_flow(image_b64, media_type, session)


async def start_receipt_flow(image_url: str, session: Session) -> str:
    """Download receipt image from URL (WhatsApp/Twilio) and start assignment flow."""
    image_b64, media_type = await _download_image(image_url)
    return await _start_flow(image_b64, media_type, session)


async def _start_flow(image_b64: str, media_type: str, session: Session) -> str:

    items = await _extract_items(image_b64, media_type)
    if not items:
        return "I couldn't read any items from the receipt. Please try a clearer photo."


    session.receipt_items = [ReceiptItem(name=i["name"], price=float(i["price"])) for i in items]
    session.current_item_index = 0
    session.mode = "receipt"

    # Pre-fetch friends for name resolution
    try:
        raw = await bridge.call_tool("get-friends", {})
        friends_data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(friends_data, dict):
            friends_data = friends_data.get("friends", [])
        session.friends = [{"id": f.get("id"), "name": _friend_name(f)} for f in (friends_data or [])]
    except Exception:
        session.friends = []

    total = sum(i.price for i in session.receipt_items)
    items_list = "\n".join(
        f"  {i + 1}. {item.name}: ${item.price:.2f}"
        for i, item in enumerate(session.receipt_items)
    )
    friends_hint = ""
    if session.friends:
        names = ", ".join(f["name"] for f in session.friends[:5])
        friends_hint = f"\nYour friends: {names}"

    first = session.receipt_items[0]
    return (
        f"🧾 Receipt items:\n{items_list}\n\nTotal: ${total:.2f}{friends_hint}\n\n"
        f"Reply *total* to split the full amount equally, "
        f"or tell me who pays for *{first.name}* (${first.price:.2f})."
    )


async def handle_assignment_response(user_response: str, session: Session) -> str:
    """Route to total-split or item-by-item flow based on user response."""

    # ── Total split shortcut ────────────────────────────────────────────────
    if session.mode == "receipt_total":
        return await _handle_total_split(user_response, session)

    # Detect "total" intent on any response during item-by-item flow
    words = set(user_response.lower().split())
    if words & _TOTAL_KEYWORDS:
        total = sum(i.price for i in session.receipt_items)
        session.mode = "receipt_total"
        friends_hint = ""
        if session.friends:
            names = ", ".join(f["name"] for f in session.friends[:5])
            friends_hint = f" (known: {names})"
        return (
            f"Got it! Who splits the total of ${total:.2f}?{friends_hint}\n"
            f"Say e.g. 'me and Monica 50/50' or 'equally among everyone in <group>'."
        )

    # ── Item-by-item flow ────────────────────────────────────────────────────
    return await _handle_item_assignment(user_response, session)


async def _handle_item_assignment(user_response: str, session: Session) -> str:
    current = session.receipt_items[session.current_item_index]
    friends_json = json.dumps([f["name"] for f in session.friends])

    prompt = f"""Parse this expense split response: "{user_response}"

Item: {current.name} (${current.price:.2f})
Known friends: {friends_json}

Return ONLY valid JSON (percentages must sum to 100):
{{"payers": [{{"name": "me", "percentage": 50}}, {{"name": "John", "percentage": 50}}]}}

Rules:
- "me", "myself", "I" → use name "me"
- "all", "just me", "only me" → 100% to "me"
- "equally" or "50/50" with two names → 50% each
- Match names to known friends (case-insensitive partial match)"""

    raw = await parse_with_haiku(prompt)
    raw = _strip_fences(raw)

    try:
        data = json.loads(raw)
        payers = data.get("payers", [])
        total_pct = sum(p["percentage"] for p in payers)
        if abs(total_pct - 100) > 1:
            return f"Percentages add up to {total_pct}%, not 100%. Who pays for *{current.name}*?"
    except json.JSONDecodeError:
        return f"I didn't understand that. Who pays for *{current.name}* (${current.price:.2f})?"

    for payer in payers:
        payer["user_id"] = _resolve_friend_id(payer["name"], session.friends)

    current.payers = payers
    session.current_item_index += 1

    if session.current_item_index < len(session.receipt_items):
        next_item = session.receipt_items[session.current_item_index]
        return f"Got it! Who pays for *{next_item.name}* (${next_item.price:.2f})?"

    return await _create_expenses(session)


async def _handle_total_split(user_response: str, session: Session) -> str:
    """Create a single expense for the receipt total with the specified split."""
    total = sum(i.price for i in session.receipt_items)
    friends_json = json.dumps([f["name"] for f in session.friends])

    prompt = f"""Parse this expense split response: "{user_response}"

Total amount: ${total:.2f}
Known friends: {friends_json}

Return ONLY valid JSON (percentages must sum to 100):
{{"payers": [{{"name": "me", "percentage": 50}}, {{"name": "Monica", "percentage": 50}}]}}

Rules:
- "me", "myself", "I" → use name "me"
- "equally" with N people → divide 100 equally
- Match names to known friends (case-insensitive)"""

    raw = await parse_with_haiku(prompt)
    raw = _strip_fences(raw)

    try:
        data = json.loads(raw)
        payers = data.get("payers", [])
        total_pct = sum(p["percentage"] for p in payers)
        if abs(total_pct - 100) > 1:
            return f"Percentages don't add up to 100%. Please re-specify the split."
    except json.JSONDecodeError:
        return "I didn't understand that. Please say who splits the total and in what percentages."

    for payer in payers:
        payer["user_id"] = _resolve_friend_id(payer["name"], session.friends)

    # Mark all items with the same payers (proportional to their price)
    for item in session.receipt_items:
        item.payers = payers

    session.mode = "receipt"  # _create_expenses resets mode to chat
    return await _create_expenses(session)


async def _create_expenses(session: Session) -> str:
    """Create Splitwise expenses for all assigned receipt items."""
    created: list[str] = []
    errors: list[str] = []
    current_user_id = await _get_current_user_id()

    for item in session.receipt_items:
        if not item.payers:
            continue
        try:
            args: dict = {
                "description": item.name,
                "cost": str(round(item.price, 2)),
                "currency_code": "USD",
            }
            if session.group_id:
                args["group_id"] = session.group_id

            users = []
            for payer in item.payers:
                share = round(item.price * payer["percentage"] / 100, 2)
                uid = payer.get("user_id") or current_user_id
                users.append({
                    "user_id": uid,
                    "paid_share": str(share),
                    "owed_share": str(share),
                })
            if users:
                args["users"] = users

            await bridge.call_tool("create-expense", args)
            created.append(item.name)
        except Exception as exc:
            logger.exception("Failed to create expense for %s", item.name)
            errors.append(f"{item.name} ({exc})")

    session.mode = "chat"
    session.receipt_items = []
    session.current_item_index = 0

    lines = [f"✅ Created {len(created)} expense(s): " + ", ".join(created)]
    if errors:
        lines.append("⚠️ Failed: " + ", ".join(errors))
    lines.append("\nWhat else can I help you with?")
    return "\n".join(lines)


async def _download_image(url: str) -> tuple[str, str]:
    auth: Optional[tuple] = None
    if settings.twilio_account_sid and settings.twilio_auth_token:
        auth = (settings.twilio_account_sid, settings.twilio_auth_token)
    async with httpx.AsyncClient() as http:
        resp = await http.get(url, auth=auth, follow_redirects=True)
        resp.raise_for_status()
        media_type = resp.headers.get("content-type", "image/jpeg").split(";")[0]
        b64 = base64.standard_b64encode(resp.content).decode()
    return b64, media_type


async def _extract_items(image_b64: str, media_type: str) -> list[dict]:
    response = _anthropic.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {
                    "type": "text",
                    "text": (
                        "Extract all line items from this receipt. "
                        "Return a JSON array: [{\"name\": \"...\", \"price\": 0.00}]. "
                        "Include tax and tip as separate items if present. "
                        "Return ONLY the JSON array, no other text."
                    ),
                },
            ],
        }],
    )
    raw = response.content[0].text.strip()
    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Failed to parse receipt items: %s", raw)
        return []


async def _get_current_user_id() -> Optional[int]:
    try:
        raw = await bridge.call_tool("get-current-user", {})
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict):
            user = data.get("user") or data
            return user.get("id")
    except Exception:
        pass
    return None


def _resolve_friend_id(name: str, friends: list[dict]) -> Optional[int]:
    if name.lower() in ("me", "myself", "i"):
        return None
    name_lower = name.lower()
    for f in friends:
        if name_lower in f["name"].lower():
            return f["id"]
    return None


def _friend_name(f: dict) -> str:
    first = f.get("first_name", "")
    last = f.get("last_name", "")
    return f"{first} {last}".strip() or f.get("email", "Unknown")


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()
