"""Project settings from `.env` and the experiment configs in `configs/`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator, model_validator
from pydantic_core import PydanticUseDefault
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

Provider = Literal["groq", "openrouter"]
Role = Literal["analyze", "plan", "code", "repair"]


class ConfigError(Exception):
    """A setting is missing or invalid; the message says which one and how to fix it."""


class Settings(BaseSettings):
    """Values read from environment variables and `.env` (names are case-insensitive).

    Variables left over from earlier experiments (GEMINI_*, GROQ_MODEL, ...) are ignored.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Provider | None = None
    llm_model: str | None = None
    llm_model_analyze: str | None = None
    llm_model_plan: str | None = None
    llm_model_code: str | None = None
    llm_model_repair: str | None = None
    llm_fallback: str | None = None

    groq_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None  # optional external tracing only (D-19)

    default_project_path: Path = Path.home() / "SRSProjects"
    db_path: Path = PROJECT_ROOT / "data" / "srs_agent.sqlite3"
    run_mode: Literal["dev", "experiment"] = "dev"

    @field_validator("*", mode="before")
    @classmethod
    def _blank_means_default(cls, value):
        # Strips stray whitespace (including Windows line endings) and treats `KEY=` as unset
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise PydanticUseDefault()
        return value

    @field_validator("default_project_path")
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(f"DEFAULT_PROJECT_PATH must be an absolute Linux path, got '{value}'")
        return value

    @field_validator("db_path")
    @classmethod
    def _relative_to_project(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @model_validator(mode="after")
    def _consistent(self) -> "Settings":
        if self.llm_provider and self.api_key_for(self.llm_provider) is None:
            name = f"{self.llm_provider.upper()}_API_KEY"
            raise ValueError(f"LLM_PROVIDER is '{self.llm_provider}' but {name} is not set")
        if self.run_mode == "experiment" and self.llm_fallback:
            raise ValueError("LLM_FALLBACK must be empty when RUN_MODE=experiment (D-28)")
        return self

    def api_key_for(self, provider: Provider) -> SecretStr | None:
        return self.groq_api_key if provider == "groq" else self.openrouter_api_key

    def model_for(self, role: Role | None = None) -> str:
        """The model id for a pipeline role: its LLM_MODEL_<ROLE> override, else LLM_MODEL."""
        model = getattr(self, f"llm_model_{role}") if role else None
        model = model or self.llm_model
        if not model:
            raise ConfigError("LLM_MODEL is not set; add it to .env (see .env.example)")
        return model


@lru_cache
def get_settings() -> Settings:
    return Settings()


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class RunConfig(BaseModel):
    """One experiment configuration. The base and memory configs differ only in `memory`."""

    model_config = ConfigDict(extra="forbid")

    name: str
    memory: MemoryConfig


def load_run_config(name_or_path: str | Path) -> RunConfig:
    """Load `configs/<name>.yaml` (e.g. "base", "memory") or a YAML file at the given path."""
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.is_file():
        raise ConfigError(f"Run config not found: {path}")
    return RunConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
