"""
main.py - StarTrace FastAPI application entry point.
"""

import sys
import os
import logging

# Ensure backend root is strictly at index 0 of sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR in sys.path:
    sys.path.remove(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.session import init_db
from app.api import analysis as analysis_router
from app.api import model as model_router
from app.api import health as health_router
from app.api import report as report_router

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("startrace.main")


def create_app() -> FastAPI:
    app = FastAPI(
        title="StarTrace API",
        description=(
            "AI-Enabled Detection and Classification of Exoplanets from TESS Light Curves. "
            "Processes NASA TESS photometric data through a Dual-Branch 1D-CNN pipeline."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json"
    )

    # CORS configuration - allow React dev server and production frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
            "*"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Include routers
    app.include_router(health_router.router)
    app.include_router(analysis_router.router)
    app.include_router(model_router.router)
    app.include_router(report_router.router)

    @app.on_event("startup")
    async def on_startup():
        logger.info("StarTrace API starting up...")
        engine, using_sqlite = init_db()
        db_type = "SQLite (local)" if using_sqlite else "PostgreSQL"
        logger.info(f"Database initialised: {db_type}")

        # Pre-load model at startup
        try:
            from ml.inference import load_model
            from app.core.config import settings
            model, meta = load_model(settings.CHECKPOINT_PATH)
            logger.info(f"Model status: {meta.get('status')}")
        except Exception as e:
            logger.warning(f"Model load warning (non-fatal): {e}")

    @app.on_event("shutdown")
    async def on_shutdown():
        logger.info("StarTrace API shutting down.")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )
