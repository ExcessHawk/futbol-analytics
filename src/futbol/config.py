"""Centralised configuration.

Every setting comes from environment variables (optionally loaded from a .env file).
Nothing else in the codebase should call os.getenv directly: import `get_settings()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# src/futbol/config.py -> project root is two levels above the package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    log_level: str
    tesseract_cmd: str = "tesseract"
    tessdata_dir: str = str(DATA_DIR / "tessdata")
    apifootball_key: str = ""

    @property
    def database_url(self) -> str:
        """SQLAlchemy URL using the psycopg (v3) driver."""
        return (
            f"postgresql+psycopg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build settings once per process (cached, like a singleton)."""
    load_dotenv(PROJECT_ROOT / ".env")  # does not override variables already set
    return Settings(
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=int(os.getenv("DB_PORT", "5450")),
        db_name=os.getenv("DB_NAME", "futbol"),
        db_user=os.getenv("DB_USER", "futbol"),
        db_password=os.getenv("DB_PASSWORD", "futbol"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        # Tesseract binary (not on PATH on Windows by default) and the folder with the
        # spa/eng language models, kept inside the project so no admin rights are needed.
        tesseract_cmd=os.getenv("TESSERACT_CMD", "tesseract"),
        tessdata_dir=os.getenv("TESSDATA_DIR", str(DATA_DIR / "tessdata")),
        # Player stats for the bet-slip analyser (free key: https://dashboard.api-football.com).
        apifootball_key=os.getenv("APIFOOTBALL_KEY", ""),
    )
