"""
database/session.py - Database session management with PostgreSQL + SQLite fallback.
Automatically switches to a local SQLite database when PostgreSQL is unreachable.
This keeps StarTrace fully functional on a student laptop with no external DB server.
"""

import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger("startrace.db")

Base = declarative_base()

_engine = None
_SessionLocal = None
_using_sqlite = False


def _make_engine(database_url: str):
    if database_url.startswith("postgresql"):
        return create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args={}
        )
    else:
        return create_engine(database_url, connect_args={"check_same_thread": False})


def init_db(database_url: str = None):
    """
    Initialises the database engine and creates all tables.
    Falls back automatically to a local SQLite file if PostgreSQL is unreachable.
    """
    global _engine, _SessionLocal, _using_sqlite

    if database_url is None:
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/startrace"
        )

    # Attempt PostgreSQL first
    if database_url.startswith("postgresql"):
        try:
            engine = _make_engine(database_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            _engine = engine
            _using_sqlite = False
            logger.info("Connected to PostgreSQL database.")
        except Exception as pg_err:
            logger.warning(f"PostgreSQL unavailable ({pg_err}). Falling back to SQLite.")
            sqlite_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "startrace.db"
            )
            sqlite_url = f"sqlite:///{sqlite_path}"
            _engine = _make_engine(sqlite_url)
            _using_sqlite = True
            logger.info(f"Using SQLite: {sqlite_path}")
    else:
        _engine = _make_engine(database_url)
        _using_sqlite = True

    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    # Import models to register them before create_all
    from . import models  # noqa
    Base.metadata.create_all(bind=_engine)
    return _engine, _using_sqlite


def get_db():
    """FastAPI dependency: yields a database session and closes it after each request."""
    global _SessionLocal
    if _SessionLocal is None:
        init_db()
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_info():
    """Returns current database type and connection status."""
    return {
        "type": "SQLite (local fallback)" if _using_sqlite else "PostgreSQL",
        "is_sqlite_fallback": _using_sqlite,
        "connected": _engine is not None
    }
