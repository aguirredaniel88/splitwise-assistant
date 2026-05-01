"""Web client API — REST endpoints for the browser chat UI."""

import base64
import logging
import uuid

from fastapi import APIRouter, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import run_agent
from .llm import AVAILABLE_MODELS, make_provider, resolve_model
from .receipt import handle_assignment_response, start_receipt_flow_b64
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session = _sessions.get(f"web:{session_id}")
    body = req.message.strip()

    if body.lower() in ("reset", "/reset", "start over"):
        _sessions.reset(f"web:{session_id}")
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
    try:
        data = await file.read()
        b64 = base64.standard_b64encode(data).decode()
        media_type = (file.content_type or "image/jpeg").split(";")[0]
        reply = await start_receipt_flow_b64(b64, media_type, session)
    except Exception as exc:
        logger.exception("Error processing uploaded image")
        reply = f"Couldn't process the image: {str(exc)[:200]}"
    return {"reply": reply, "session_id": session_id}


@router.get("/chat/models")
async def get_models(session_id: str | None = None):
    current = "claude-sonnet-4-6"
    if session_id:
        s = _sessions.get(f"web:{session_id}")
        if s.llm_provider:
            current = s.llm_provider.name
    return {"current": current, "available": sorted(AVAILABLE_MODELS)}


@router.post("/chat/model")
async def set_model(req: ModelRequest):
    session = _sessions.get(f"web:{req.session_id}")
    resolved = resolve_model(req.model)
    if not resolved:
        return {"ok": False, "error": f"Unknown model '{req.model}'"}
    provider_name, model_id = resolved
    session.llm_provider = make_provider(provider_name, model_id)
    session.history = []
    return {"ok": True, "model": model_id}


@router.delete("/chat")
async def reset_chat(session_id: str):
    _sessions.reset(f"web:{session_id}")
    return {"ok": True}


def _model_command(body: str, session) -> str:
    parts = body.strip().split(None, 1)
    current = session.llm_provider.name if session.llm_provider else "unknown"
    if len(parts) == 1:
        return f"Current model: {current}\n\nAvailable: {', '.join(sorted(AVAILABLE_MODELS))}"
    alias = parts[1].strip()
    resolved = resolve_model(alias)
    if not resolved:
        return f"Unknown model '{alias}'. Available: {', '.join(sorted(AVAILABLE_MODELS))}"
    provider_name, model_id = resolved
    session.llm_provider = make_provider(provider_name, model_id)
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

  header { display: flex; align-items: center; gap: 12px; padding: 14px 20px;
           background: var(--surface); border-bottom: 1px solid var(--border);
           flex-shrink: 0; }
  header h1 { font-size: 1.1rem; font-weight: 600; flex: 1; }
  header h1 span { color: var(--accent); }

  select, button { font-family: var(--font); font-size: .85rem; cursor: pointer;
                   border: 1px solid var(--border); border-radius: 8px;
                   background: var(--surface2); color: var(--text); padding: 6px 12px;
                   transition: background .15s; }
  select:hover, button:hover { background: var(--border); }

  #reset-btn { color: #f47; border-color: #f47; }

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
  #img-btn { padding: 10px 12px; font-size: 1.1rem; border-radius: 12px; }
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
  <select id="model-select" title="Switch AI model"></select>
  <button id="reset-btn" title="Start a new conversation">New chat</button>
</header>

<div id="messages"></div>

<div class="input-row">
  <button id="img-btn" title="Upload a receipt">📷</button>
  <input type="file" id="file-input" accept="image/*">
  <textarea id="msg-input" rows="1" placeholder="Ask about your expenses…"></textarea>
  <button id="send-btn">Send</button>
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
    addMsg('bot', data.reply);
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
  if (sessionId) await fetch(`${API}/chat?session_id=${sessionId}`, {method: 'DELETE'});
  document.getElementById('messages').innerHTML = '';
  addMsg('bot', 'New conversation started. How can I help you?');
});

// ── Init ─────────────────────────────────────────────────────────────────────
loadModels();
addMsg('bot', 'Hi! Ask me anything about your Splitwise expenses, or drop a receipt photo to split it.');
</script>
</body>
</html>"""
