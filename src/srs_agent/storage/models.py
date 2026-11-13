"""Database tables for run records. All times are stored in UTC."""

from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel

# Bump when a table or column changes; an older database file then fails with a clear error
SCHEMA_VERSION = 1


def utcnow() -> datetime:
    return datetime.now(UTC)


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    needs_clarification = "needs_clarification"
    done = "done"
    failed = "failed"


class RunStatus(StrEnum):
    running = "running"
    passed = "passed"
    failed = "failed"
    needs_clarification = "needs_clarification"
    infra_error = "infra_error"


class AttemptKind(StrEnum):
    generate = "generate"
    repair = "repair"


class TestSource(StrEnum):
    __test__ = False

    agent = "agent"  # the developer agent's own tests
    hidden = "hidden"  # the evaluation task's hidden acceptance tests


class TestOutcome(StrEnum):
    __test__ = False

    passed = "passed"
    failed = "failed"
    error = "error"
    skipped = "skipped"


class CallStatus(StrEnum):
    ok = "ok"
    error = "error"


class SchemaInfo(SQLModel, table=True):
    __tablename__ = "schema_info"

    id: int = Field(default=1, primary_key=True)
    version: int


class Job(SQLModel, table=True):
    """One uploaded SRS."""

    __tablename__ = "jobs"

    id: int | None = Field(default=None, primary_key=True)
    srs_filename: str
    srs_sha256: str = Field(index=True)
    status: JobStatus = JobStatus.queued
    app_category: str | None = None
    project_dir: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class Run(SQLModel, table=True):
    """One run of the agent on a job, under one configuration."""

    __tablename__ = "runs"

    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="jobs.id", index=True)
    config_name: str  # "base" or "memory"
    memory_enabled: bool
    run_mode: str  # "dev" or "experiment"
    seed: int | None = None
    provider: str | None = None
    model: str | None = None
    status: RunStatus = RunStatus.running
    attempts_used: int = 0
    error: str | None = None
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None


class Attempt(SQLModel, table=True):
    """One generate or repair cycle within a run (number 0 is the first generate)."""

    __tablename__ = "attempts"
    __table_args__ = (UniqueConstraint("run_id", "number"),)

    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="runs.id", index=True)
    number: int
    kind: AttemptKind
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None


class TestResult(SQLModel, table=True):
    """One lint and test session after an attempt."""

    __test__ = False
    __tablename__ = "test_results"

    id: int | None = Field(default=None, primary_key=True)
    attempt_id: int = Field(foreign_key="attempts.id", index=True)
    source: TestSource
    lint_errors: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    duration_s: float | None = None
    failure_summary: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class TestCase(SQLModel, table=True):
    """One test within a session, with the requirement it checks (for requirement coverage)."""

    __test__ = False
    __tablename__ = "test_cases"

    id: int | None = Field(default=None, primary_key=True)
    test_result_id: int = Field(foreign_key="test_results.id", index=True)
    nodeid: str
    requirement_id: str | None = Field(default=None, index=True)
    outcome: TestOutcome


class LLMCall(SQLModel, table=True):
    """One call to a language model. `run_id` is empty for calls outside a run (benchmarks)."""

    __tablename__ = "llm_calls"

    id: int | None = Field(default=None, primary_key=True)
    run_id: int | None = Field(default=None, foreign_key="runs.id", index=True)
    node: str | None = None  # pipeline stage, e.g. "analyze"
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    retries: int = 0
    status: CallStatus = CallStatus.ok
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
