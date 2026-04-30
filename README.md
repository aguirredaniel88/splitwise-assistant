# Splitwise Assistant v2

A WhatsApp chatbot that manages Splitwise expenses using natural language and receipt image recognition.

## Features

- **Natural language chat** — ask about balances, create expenses, manage groups
- **WhatsApp interface** — send messages via Twilio WhatsApp
- **Receipt scanning** — photo a receipt and it extracts each item, then walks you through who pays what (with percentages)
- **47 Splitwise tools** powered by the [splitwise-mcp](../splitwise-mcp) FastMCP server

## Architecture

```
WhatsApp (Twilio) → FastAPI webhook → Claude Sonnet (tool_use) → FastMCP Client → splitwise-mcp → Splitwise API
                                    ↕
                              Session state (in-memory)
```

## Local Development

### 1. Prerequisites

```bash
# Install splitwise-mcp (sibling directory)
pip install -e ../splitwise-mcp

# Install this app
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, SPLITWISE_OAUTH_ACCESS_TOKEN,
# TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
# Set SPLITWISE_MCP_PATH to the absolute path of splitwise-mcp/app.py
```

### 3. Run

```bash
python -m splitwise_assistant.main
# or: uvicorn splitwise_assistant.main:app --reload
```

### 4. Expose for Twilio webhook

```bash
ngrok http 8000
# Copy the HTTPS URL and paste it into Twilio Console:
# Messaging → Active numbers → your WhatsApp number → Webhook URL:
#   https://<ngrok-id>.ngrok.io/webhook/whatsapp  (HTTP POST)
```

## Deploy to Render (free)

### 1. Prepare the repo

The Dockerfile expects the `splitwise-mcp` directory to be copied in at build time.
Either add the MCP as a git submodule, or copy it into the repo:

```bash
# Option A: git submodule (recommended)
git submodule add <splitwise-mcp-repo-url> splitwise-mcp

# Option B: copy
cp -r ../splitwise-mcp ./splitwise-mcp
git add splitwise-mcp
```

### 2. Deploy

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render detects `render.yaml` automatically
5. Add your secret env vars in the Render dashboard:
   - `ANTHROPIC_API_KEY`
   - `SPLITWISE_OAUTH_ACCESS_TOKEN`
   - `TWILIO_ACCOUNT_SID`
   - `TWILIO_AUTH_TOKEN`
6. Deploy — your webhook URL will be `https://<service>.onrender.com/webhook/whatsapp`

### 3. Configure Twilio

In [Twilio Console](https://console.twilio.com):
- Messaging → Settings → WhatsApp Sandbox (for dev) or your production number
- Set webhook URL to `https://<your-render-url>/webhook/whatsapp`

## WhatsApp Usage

**Chat examples:**
```
"What do I owe?"
"Create a $45 dinner expense split equally with Alice and Bob in the Barcelona trip group"
"Show my recent expenses"
"How much does John owe me?"
```

**Receipt flow:**
1. Send a photo of any receipt
2. The bot extracts all items and shows them
3. For each item, reply with who pays (e.g. "Me and Sarah 50/50")
4. After the last item, expenses are created automatically in Splitwise

**Commands:**
- `reset` — clear conversation and start over

## Splitwise OAuth Setup

```bash
cd ../splitwise-mcp
python -m splitwise_mcp_server.oauth_setup
```

Follow the prompts to get your `SPLITWISE_OAUTH_ACCESS_TOKEN`.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Your Anthropic API key |
| `SPLITWISE_MCP_PATH` | ✅ | Absolute path to `splitwise-mcp/app.py` |
| `SPLITWISE_OAUTH_ACCESS_TOKEN` | ✅* | Splitwise OAuth token |
| `SPLITWISE_API_KEY` | ✅* | Alternative: Splitwise API key |
| `TWILIO_ACCOUNT_SID` | ✅ | Twilio account SID |
| `TWILIO_AUTH_TOKEN` | ✅ | Twilio auth token |
| `TWILIO_WHATSAPP_NUMBER` | | Defaults to sandbox number |
| `SESSION_TTL_MINUTES` | | Session timeout (default: 60) |
| `PORT` | | Server port (default: 8000) |

*One of OAuth token or API key required.
