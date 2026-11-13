"""SQLite engine, sessions and schema setup."""

from pathlib import Path

from sqlalchemy import Engine, event, inspect, text
from sqlmodel import Session, SQLModel, create_engine

from srs_agent.config import ConfigError, get_settings
from srs_agent.storage.models import SCHEMA_VERSION, SchemaInfo


def make_engine(path: Path | None = None) -> Engine:
    """Engine for the SQLite file at `path` (default: DB_PATH), with WAL and foreign keys on."""
    path = path or get_settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}")

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    """Create any missing tables, refusing a database written by a different schema version."""
    if inspect(engine).has_table(SchemaInfo.__tablename__):
        with engine.connect() as conn:
            found = conn.execute(text("SELECT version FROM schema_info WHERE id = 1")).scalar()
        if found is not None and found != SCHEMA_VERSION:
            raise ConfigError(
                f"Database {engine.url.database} has schema version {found} but the code "
                f"expects {SCHEMA_VERSION}. In development, delete the file and run again."
            )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        if session.get(SchemaInfo, 1) is None:
            session.add(SchemaInfo(id=1, version=SCHEMA_VERSION))
            session.commit()


def open_session(engine: Engine) -> Session:
    """A session for reading and writing records; use it as a `with` block and commit yourself."""
    return Session(engine)
