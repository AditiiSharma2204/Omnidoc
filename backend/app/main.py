from fastapi import FastAPI

from app.config.settings import settings
from app.api.v1.health import router as health_router
from app.api.v1.documents import router as document_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise Multimodal Document Intelligence Platform",
)

# Health Routes
app.include_router(
    health_router,
    prefix=settings.API_PREFIX,
    tags=["Health"],
)

# Document Routes
app.include_router(
    document_router,
    prefix=f"{settings.API_PREFIX}/documents",
    tags=["Documents"],
)


@app.get("/")
def root():
    return {
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }