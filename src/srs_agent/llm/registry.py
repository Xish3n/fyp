"""Model registry: the only place model ids and their capabilities are listed (D-31)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from srs_agent.config import CONFIG_DIR, ConfigError, Provider, Role, Settings

REGISTRY_PATH = CONFIG_DIR / "models.yaml"


class ModelSpec(BaseModel):
    """One model on one provider. `None` means not yet measured (filled in by task 0.5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    provider: Provider
    supports_tools: bool | None = None
    supports_json_schema: bool | None = None
    context_window: int | None = None
    rpm: int | None = None  # requests per minute
    tpm: int | None = None  # tokens per minute
    price_in: float | None = None  # USD per 1M input tokens; 0 for free models
    price_out: float | None = None  # USD per 1M output tokens
    notes: str = ""

    @field_validator("id")
    @classmethod
    def _no_openai_api_models(cls, value: str) -> str:
        # OpenAI models are excluded (D-18); its open-weight gpt-oss models are allowed (D-44)
        if value.startswith("openai/") and "gpt-oss" not in value:
            raise ValueError(
                f"'{value}' is an OpenAI API model; only open-weight gpt-oss is allowed"
            )
        return value


class ModelRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelSpec]

    @model_validator(mode="after")
    def _unique(self) -> "ModelRegistry":
        seen = set()
        for m in self.models:
            key = (m.provider, m.id)
            if key in seen:
                raise ValueError(f"Model '{m.id}' is listed twice for provider '{m.provider}'")
            seen.add(key)
        return self

    def get(self, model_id: str, provider: Provider) -> ModelSpec:
        for m in self.models:
            if m.id == model_id and m.provider == provider:
                return m
        known = ", ".join(m.id for m in self.models if m.provider == provider) or "none yet"
        raise ConfigError(
            f"Model '{model_id}' is not in configs/models.yaml for provider '{provider}' "
            f"(listed: {known})"
        )


def load_registry(path: Path = REGISTRY_PATH) -> ModelRegistry:
    if not path.is_file():
        raise ConfigError(f"Model registry not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ModelRegistry.model_validate({"models": data.get("models") or []})


def resolve_model(
    settings: Settings, role: Role | None = None, registry: ModelRegistry | None = None
) -> ModelSpec:
    """The registry entry for the model this role should use, per `.env`."""
    if not settings.llm_provider:
        raise ConfigError("LLM_PROVIDER is not set; use 'groq' or 'openrouter' in .env")
    registry = registry or load_registry()
    return registry.get(settings.model_for(role), settings.llm_provider)
