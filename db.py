"""Database plumbing.

The app's whole surface on Postgres is the SQL functions in supabase/migrations.

    uv run db.py migrate    apply the migrations to a new database
    uv run db.py reset      drop the vetvoice schema, migrate again and load the seed
"""

import os
import sys
from pathlib import Path

import psycopg
from psycopg.conninfo import conninfo_to_dict

ROOT = Path(__file__).parent
DSN = os.environ.get("DATABASE_URL", "postgresql://localhost/vetvoice")


def _run(dsn: str, *paths: Path) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        for path in paths:
            conn.execute(path.read_text(encoding="utf-8"))


def migrate(dsn: str = DSN) -> None:
    _run(dsn, *sorted((ROOT / "supabase" / "migrations").glob("*.sql")))


def reset(dsn: str = DSN, seed: bool = True) -> None:
    """Drop and rebuild the vetvoice schema. Every call and appointment in it is lost."""
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("drop schema if exists vetvoice cascade")
    migrate(dsn)
    if seed:
        _run(dsn, ROOT / "supabase" / "seed.sql")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "migrate":
        migrate()
    elif command == "reset":
        host = conninfo_to_dict(DSN).get("host") or "localhost"
        if host not in ("localhost", "127.0.0.1") and "--force" not in sys.argv:
            sys.exit(f"refusing to reset {host}: pass --force if you really mean it")
        reset()
    else:
        sys.exit(__doc__)
