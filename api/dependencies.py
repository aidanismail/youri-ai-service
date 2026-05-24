from __future__ import annotations

import logging
import os
import secrets
from functools import lru_cache
from typing import Annotated

from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from ml_engine.information_retrieval import LexicalMatcherModel, SmartSubstitutionEngine

API_KEY_NAME = "x-internal-api-key"
DEFAULT_INTERNAL_KEY = "youri-super-secret-key-2026"

LOGGER = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def _load_internal_api_key() -> str:
    api_key = os.getenv("INTERNAL_API_KEY")
    if api_key:
        return api_key

    LOGGER.warning("INTERNAL_API_KEY belum diset; memakai default development key.")
    return DEFAULT_INTERNAL_KEY


SECRET_INTERNAL_KEY = _load_internal_api_key()


async def verify_internal_api_key(
    api_key: Annotated[str | None, Security(api_key_header)],
) -> str:
    if not api_key or not secrets.compare_digest(api_key, SECRET_INTERNAL_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Akses ditolak: internal API key tidak valid atau hilang.",
        )

    return api_key


@lru_cache(maxsize=1)
def _get_lexical_matcher() -> LexicalMatcherModel:
    return LexicalMatcherModel()


def get_lexical_matcher() -> LexicalMatcherModel:
    model = _get_lexical_matcher()
    if not model.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=model.load_error or "Lexical matcher belum siap.",
        )

    return model


@lru_cache(maxsize=1)
def _get_substitution_engine() -> SmartSubstitutionEngine:
    return SmartSubstitutionEngine()


def get_substitution_engine() -> SmartSubstitutionEngine:
    engine = _get_substitution_engine()
    if not engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=engine.load_error or "Substitution engine belum siap.",
        )

    return engine
