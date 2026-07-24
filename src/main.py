"""App entrypoint. The gateway (and its provider clients) is built once at
startup and stashed on app.state, so every request reuses the same warm
connection pools rather than reconstructing clients per call."""

from __future__ import annotations
from fastapi import FastAPI
from contextlib import asynccontextmanager


from api.chat import router as chat_router
from api.health import router as health_router
from core.config import build_gateway, get_settings
from core.log import get_logger
from middlewares.request_id import RequestMiddleware

logger = get_logger("Resilient-LLM-Gateway")

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.gateway = build_gateway(settings)
    logger.info(
        "gateway ready with chain: %s", app.state.gateway.chain_labels
    )
    yield


app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)

app.add_middleware(RequestMiddleware)

app.include_router(health_router)
app.include_router(chat_router)


if __name__ == "__main__":
    # Lets you run the app directly: `python src/main.py`.
    # Running the file puts its own dir (src/) on sys.path, so the absolute
    # imports above (core.*, api.*) resolve without any PYTHONPATH setup.
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
    