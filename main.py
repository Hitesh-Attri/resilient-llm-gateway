"""App entrypoint. The gateway (and its provider clients) is built once at
startup and stashed on app.state, so every request reuses the same warm
connection pools rather than reconstructing clients per call."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.chat import router as chat_router
from .api.health import router as health_router
from .core.config import build_gateway, get_settings

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.gateway = build_gateway(settings)
    logging.getLogger("llm_gateway").info(
        "gateway ready with chain: %s", app.state.gateway.chain_labels
    )
    yield


app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)
app.include_router(health_router)
app.include_router(chat_router)
