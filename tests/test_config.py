"""Tests for configuration loading."""

from futbol.config import PROJECT_ROOT, Settings, get_settings


def test_database_url_uses_psycopg_driver():
    s = Settings("h", 1234, "db", "u", "p", "INFO")
    assert s.database_url == "postgresql+psycopg://u:p@h:1234/db"


def test_env_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("DB_PORT", "6000")
    get_settings.cache_clear()  # settings are cached; reset to re-read the env
    try:
        assert get_settings().db_port == 6000
    finally:
        get_settings.cache_clear()


def test_project_root_points_at_repo():
    assert (PROJECT_ROOT / "pyproject.toml").exists()
