from __future__ import annotations

import logging

from fastapi import FastAPI

from api.routers import cooking

APP_TITLE = "Youri API - ML Engine Service"
APP_VERSION = "1.0.0"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="Sistem AI/ML internal untuk Youri Smart Cooking Assistant.",
)

app.include_router(cooking.router, prefix="/v1/ai/cooking")


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": APP_TITLE,
        "version": APP_VERSION,
    }
