"""Database access: one shared SQLAlchemy engine plus bulk-load helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sqlalchemy import Connection, Engine, create_engine, text

from futbol.config import PROJECT_ROOT, get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide engine (it owns the connection pool).

    pool_pre_ping checks a connection before handing it out, so a dropped
    connection (DB restart, idle timeout on a hosted DB) is replaced silently.
    connect_timeout makes an unreachable server fail fast instead of hanging.
    """
    return create_engine(
        get_settings().database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )


def ping() -> str:
    """Run a trivial query and return the server version. Raises if unreachable."""
    with get_engine().connect() as conn:
        return conn.execute(text("SELECT version()")).scalar_one()


def run_sql_file(path: Path | str) -> None:
    """Execute a .sql file (several statements) in one transaction."""
    sql = Path(path).read_text(encoding="utf-8")
    with get_engine().begin() as conn:
        # Straight to psycopg without parameters, so '%' in comments isn't a placeholder.
        conn.connection.driver_connection.execute(sql)


def init_schema() -> None:
    run_sql_file(PROJECT_ROOT / "sql" / "schema.sql")


def _to_python_rows(df: pd.DataFrame) -> Iterable[tuple]:
    """Yield rows with NaN/NA converted to None (what the DB driver expects)."""
    clean = df.astype(object).where(df.notna(), None)
    return clean.itertuples(index=False, name=None)


def copy_rows(conn: Connection, table: str, df: pd.DataFrame) -> int:
    """Bulk-insert a DataFrame with PostgreSQL COPY (10-50x faster than INSERT).

    Column names in `df` must match the table. Runs inside the caller's transaction.
    """
    if df.empty:
        return 0
    cols = ", ".join(df.columns)
    raw = conn.connection.driver_connection  # the underlying psycopg connection
    with raw.cursor() as cur, cur.copy(f"COPY {table} ({cols}) FROM STDIN") as copy:
        for row in _to_python_rows(df):
            copy.write_row(row)
    return len(df)


def upsert_rows(
    conn: Connection,
    table: str,
    df: pd.DataFrame,
    key: Sequence[str],
    update: bool = True,
) -> int:
    """INSERT ... ON CONFLICT for small/medium tables (teams, matches, odds).

    With update=True existing rows get their non-key columns overwritten, which is
    what makes re-running an ingestion idempotent instead of failing or duplicating.
    """
    if df.empty:
        return 0
    cols = list(df.columns)
    non_key = [c for c in cols if c not in key]
    conflict = (
        f"DO UPDATE SET {', '.join(f'{c} = EXCLUDED.{c}' for c in non_key)}"
        if update and non_key
        else "DO NOTHING"
    )
    stmt = text(
        f"INSERT INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join(':' + c for c in cols)}) "
        f"ON CONFLICT ({', '.join(key)}) {conflict}"
    )
    records = [dict(zip(cols, row, strict=True)) for row in _to_python_rows(df)]
    conn.execute(stmt, records)
    return len(records)


def read_sql(query: str, **params) -> pd.DataFrame:
    """Run a SELECT with bound parameters and return a DataFrame."""
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn, params=params)
