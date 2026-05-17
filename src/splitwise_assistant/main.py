"""FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .config import settings
from .mcp_bridge import create_bridge, MCPBridge
from .web import router as web_router
from .whatsapp import router as whatsapp_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# Global bridge for WhatsApp endpoint (backward compatibility)
_global_bridge: Optional[MCPBridge] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _global_bridge

    # Create default bridge from environment for WhatsApp and health check
    _global_bridge = create_bridge(
        oauth_access_token=settings.splitwise_oauth_access_token,
        api_key=settings.splitwise_api_key
    )
    await _global_bridge.startup()
    logger.info("Global bridge initialized for WhatsApp endpoint")

    yield

    if _global_bridge:
        await _global_bridge.shutdown()


app = FastAPI(title="Splitwise Assistant", version="0.1.0", lifespan=lifespan)

# Mount static files for PWA
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/api/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(whatsapp_router)
app.include_router(web_router)


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    if _global_bridge:
        tools = await _global_bridge.list_tools()
        return {"status": "ok", "tools_available": len(tools)}
    return {"status": "ok", "tools_available": 0}


@app.get("/")
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/")


def main():
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("splitwise_assistant.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
