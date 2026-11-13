import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from srs_agent.config import ConfigError
from srs_agent.storage.db import init_db, make_engine, open_session
from srs_agent.storage.models import (
    SCHEMA_VERSION,
    Attempt,
    AttemptKind,
    Job,
    LLMCall,
    Run,
    TestCase,
    TestOutcome,
    TestResult,
    TestSource,
)

TABLES = {"schema_info", "jobs", "runs", "attempts", "test_results", "test_cases", "llm_calls"}


@pytest.fixture
def engine(tmp_path):
    engine = make_engine(tmp_path / "test.sqlite3")
    init_db(engine)
    yield engine
    engine.dispose()


def test_creates_all_tables_with_wal(engine):
    assert set(inspect(engine).get_table_names()) >= TABLES
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


def test_creates_missing_parent_folder(tmp_path):
    engine = make_engine(tmp_path / "nested" / "dir" / "db.sqlite3")
    init_db(engine)
    assert (tmp_path / "nested" / "dir" / "db.sqlite3").is_file()
    engine.dispose()


def test_full_chain_round_trip(engine):
    with open_session(engine) as s:
        job = Job(srs_filename="calc.pdf", srs_sha256="abc")
        s.add(job)
        s.commit()
        run = Run(job_id=job.id, config_name="memory", memory_enabled=True, run_mode="dev")
        s.add(run)
        s.commit()
        attempt = Attempt(run_id=run.id, number=0, kind=AttemptKind.generate)
        s.add(attempt)
        s.commit()
        result = TestResult(attempt_id=attempt.id, source=TestSource.hidden, passed=1, failed=1)
        s.add(result)
        s.commit()
        s.add(
            TestCase(
                test_result_id=result.id,
                nodeid="t::a",
                requirement_id="FR-01",
                outcome=TestOutcome.passed,
            )
        )
        s.add(
            TestCase(
                test_result_id=result.id,
                nodeid="t::b",
                requirement_id="FR-02",
                outcome=TestOutcome.failed,
            )
        )
        s.commit()

    with open_session(engine) as s:
        cases = s.exec(select(TestCase).order_by(TestCase.nodeid)).all()
        assert [(c.requirement_id, c.outcome) for c in cases] == [
            ("FR-01", TestOutcome.passed),
            ("FR-02", TestOutcome.failed),
        ]
        assert s.exec(select(Run)).one().memory_enabled is True


def test_broken_links_are_refused(engine):
    with open_session(engine) as s:
        s.add(Run(job_id=999, config_name="base", memory_enabled=False, run_mode="dev"))
        with pytest.raises(IntegrityError):
            s.commit()


def test_attempt_numbers_are_unique_per_run(engine):
    with open_session(engine) as s:
        job = Job(srs_filename="a.pdf", srs_sha256="x")
        s.add(job)
        s.commit()
        run = Run(job_id=job.id, config_name="base", memory_enabled=False, run_mode="dev")
        s.add(run)
        s.commit()
        s.add(Attempt(run_id=run.id, number=0, kind=AttemptKind.generate))
        s.add(Attempt(run_id=run.id, number=0, kind=AttemptKind.repair))
        with pytest.raises(IntegrityError):
            s.commit()


def test_llm_call_without_run(engine):
    with open_session(engine) as s:
        s.add(LLMCall(provider="groq", model="m1", prompt_tokens=10, completion_tokens=5))
        s.commit()
        call = s.exec(select(LLMCall)).one()
        assert call.run_id is None and call.retries == 0


def test_init_db_is_safe_to_repeat(engine):
    init_db(engine)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT version FROM schema_info")).scalar() == SCHEMA_VERSION


def test_old_schema_version_is_a_clear_error(engine):
    with engine.begin() as conn:
        conn.execute(text("UPDATE schema_info SET version = 0"))
    with pytest.raises(ConfigError, match="schema version 0"):
        init_db(engine)


def test_db_path_default_and_relative(make_settings):
    from srs_agent.config import PROJECT_ROOT

    assert make_settings().db_path == PROJECT_ROOT / "data" / "srs_agent.sqlite3"
    assert (
        make_settings(db_path="data/other.sqlite3").db_path == PROJECT_ROOT / "data/other.sqlite3"
    )
