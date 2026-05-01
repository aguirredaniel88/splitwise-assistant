"""FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .mcp_bridge import bridge
from .web import router as web_router
from .whatsapp import router as whatsapp_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await bridge.startup()
    yield
    await bridge.shutdown()


app = FastAPI(title="Splitwise Assistant", version="0.1.0", lifespan=lifespan)
app.include_router(whatsapp_router)
app.include_router(web_router)


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    tools = await bridge.list_tools()
    return {"status": "ok", "tools_available": len(tools)}


@app.get("/")
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/")


def main():
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("splitwise_assistant.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
