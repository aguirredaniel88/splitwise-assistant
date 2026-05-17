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
            openai_api_key=req.openai_api_key
        )
        return {"ok": True, "message": "Credentials validated successfully"}
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
    has_llm = bool(session.anthropic_api_key or session.openai_api_key)
    has_bridge = session.mcp_bridge is not None

    tools_count = 0
    if has_bridge:
        try:
            tools_count = len(await session.mcp_bridge.list_tools())
        except:
            pass

    return {
        "configured": has_splitwise and has_llm,
        "ready": has_bridge,
        "tools_available": tools_count
    }


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
    await _sessions.reset(f"web:{session_id}")
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
  .payer-chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .payer-chip { padding: 4px 12px; border-radius: 20px; font-size: .82rem;
                border: 1px solid var(--border); background: var(--surface2);
                cursor: pointer; transition: background .15s, border-color .15s; color: var(--text); }
  .payer-chip:hover { background: var(--border); }
  .payer-chip.selected { background: var(--accent); border-color: var(--accent); color: #fff; }
  .receipt-divider { border: none; border-top: 1px solid var(--border); margin: 14px 0; }
  .receipt-submit { width: 100%; padding: 9px; background: var(--accent);
                    border-color: var(--accent); color: #fff; border-radius: 10px;
                    font-weight: 600; font-size: .9rem; margin-top: 14px; }
  .receipt-submit:hover:not(:disabled) { background: var(--accent-h); border-color: var(--accent-h); }
  .receipt-submit:disabled { opacity: .45; cursor: not-allowed; }

  /* ── Credentials modal ─────────────────────────────────────────────────── */
  .credentials-modal {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.85);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal-content {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    max-width: 500px;
    width: 90%;
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
  #img-btn, #mic-btn { padding: 10px 12px; font-size: 1.1rem; border-radius: 12px; }
  #mic-btn.recording { background: #f47; border-color: #f47; color: #fff; animation: pulse 1.5s infinite; }
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
  <select id="model-select" title="Switch AI model"></select>
  <button id="reset-btn" title="Start a new conversation">New chat</button>
</header>

<div id="messages"></div>

<div class="input-row">
  <button id="img-btn" title="Upload a receipt">📷</button>
  <input type="file" id="file-input" accept="image/*">
  <button id="mic-btn" title="Voice input">🎤</button>
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
        • Anthropic: <a href="https://console.anthropic.com/account/keys" target="_blank" style="color: var(--accent);">console.anthropic.com</a><br>
        • OpenAI: <a href="https://platform.openai.com/api-keys" target="_blank" style="color: var(--accent);">platform.openai.com</a><br>
        • Splitwise: <a href="https://secure.splitwise.com/apps" target="_blank" style="color: var(--accent);">secure.splitwise.com/apps</a>
      </p>

      <h3 style="margin-top: 20px; margin-bottom: 8px; font-size: 0.95rem;">LLM Provider (required)</h3>
      <label for="anthropic-key">Anthropic API Key</label>
      <input type="password" id="anthropic-key" placeholder="sk-ant-...">

      <div class="divider">OR</div>

      <label for="openai-key">OpenAI API Key</label>
      <input type="password" id="openai-key" placeholder="sk-...">

      <h3 style="margin-top: 24px; margin-bottom: 8px; font-size: 0.95rem;">Splitwise (required)</h3>
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
  const anthropicKey = document.getElementById('anthropic-key').value.trim();
  const openaiKey = document.getElementById('openai-key').value.trim();
  const oauthToken = document.getElementById('oauth-token').value.trim();
  const apiKey = document.getElementById('api-key').value.trim();
  const errorDiv = document.getElementById('cred-error');

  // Validate inputs
  const hasLLM = anthropicKey || openaiKey;
  const hasSplitwise = oauthToken || apiKey;

  if (!hasLLM) {
    errorDiv.textContent = 'Please provide Anthropic or OpenAI API key';
    errorDiv.style.display = 'block';
    return;
  }

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
        anthropic_api_key: anthropicKey || null,
        openai_api_key: openaiKey || null,
        oauth_token: oauthToken || null,
        api_key: apiKey || null
      }));

      document.getElementById('creds-modal').remove();
      addMsg('bot', '✅ Connected! How can I help you with your Splitwise expenses?');
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
    return data.ok;
  } catch {
    return false;
  }
}

// ── Voice Input ──────────────────────────────────────────────────────────────
let recognition = null;
let isRecording = false;

function initVoiceInput() {
  // Check for browser support
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.warn('Speech recognition not supported in this browser');
    document.getElementById('mic-btn').style.display = 'none';
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    isRecording = true;
    document.getElementById('mic-btn').classList.add('recording');
    document.getElementById('msg-input').placeholder = 'Listening...';
  };

  recognition.onend = () => {
    isRecording = false;
    document.getElementById('mic-btn').classList.remove('recording');
    document.getElementById('msg-input').placeholder = 'Ask about your expenses…';
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
    document.getElementById('msg-input').placeholder = 'Ask about your expenses…';

    if (event.error === 'not-allowed') {
      addMsg('bot', '⚠️ Microphone access denied. Please allow microphone access in your browser settings.');
    }
  };
}

document.getElementById('mic-btn').addEventListener('click', () => {
  if (!recognition) {
    addMsg('bot', '⚠️ Voice input not supported in this browser. Try Chrome, Safari, or Edge.');
    return;
  }

  if (isRecording) {
    recognition.stop();
  } else {
    recognition.start();
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
    addMsg('bot', 'Hi! Ask me anything about your Splitwise expenses, or drop a receipt photo.');
  }
})();
</script>
</body>
</html>"""
