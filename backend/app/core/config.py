"""
config.py - Application configuration and environment settings for StarTrace.
"""

import os
from typing import List


class Settings:
    PROJECT_NAME: str = "StarTrace"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    # Database connection string
    # Defaults to PostgreSQL, but session.py will fallback cleanly to SQLite if PostgreSQL is unreachable
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/startrace"
    )

    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*"
    ]

    # Storage paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    STORAGE_DIR: str = os.path.join(BASE_DIR, "storage")
    SAMPLE_DATA_DIR: str = os.path.join(BASE_DIR, "sample_data")
    CHECKPOINT_PATH: str = os.path.join(BASE_DIR, "ml", "checkpoints", "weights.pth")


settings = Settings()
os.makedirs(settings.STORAGE_DIR, exist_ok=True)
