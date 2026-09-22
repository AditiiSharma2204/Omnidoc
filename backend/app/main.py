from app.bootstrap import apply_local_first_env_fixes

# Must run before ANYTHING else is imported -- docling and
# sentence-transformers both pull in huggingface_hub transitively,
# and it reads these env vars once at import time. See
# app/bootstrap.py for why this exists.
apply_local_first_env_fixes()

import logging  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.config.settings import settings
from app.api.v1.health import router as health_router
from app.api.v1.documents import router as document_router
from app.api.v1.retrieval import router as retrieval_router
from app.api.v1.chat import router as chat_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omnidoc")

# ---------------------------------------------------------
# Create required storage directories
# ---------------------------------------------------------
Path(settings.DOCUMENTS_DIR).mkdir(
    parents=True,
    exist_ok=True,
)

Path(settings.VECTORSTORE_DIR).mkdir(
    parents=True,
    exist_ok=True,
)

logger.info("DOCUMENTS_DIR=%s", settings.DOCUMENTS_DIR)
logger.info("VECTORSTORE_DIR=%s", settings.VECTORSTORE_DIR)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Local-first document intelligence platform (RAG over PDF/DOCX/PPTX/XLSX).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    health_router,
    prefix=settings.API_PREFIX,
    tags=["Health"],
)

app.include_router(
    document_router,
    prefix=f"{settings.API_PREFIX}/documents",
    tags=["Documents"],
)

app.include_router(
    retrieval_router,
    prefix=settings.API_PREFIX,
    tags=["Semantic Search"],
)

app.include_router(
    chat_router,
    prefix=settings.API_PREFIX,
    tags=["Chat"],
)


@app.get("/")
def root():
    return {
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }
