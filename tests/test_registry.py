import pytest
from pydantic import ValidationError

from srs_agent.config import ConfigError
from srs_agent.llm.registry import ModelRegistry, load_registry, resolve_model

REGISTRY = ModelRegistry.model_validate(
    {
        "models": [
            {"id": "general", "provider": "groq", "supports_tools": True, "price_in": 0},
            {"id": "coder", "provider": "groq"},
            {"id": "general", "provider": "openrouter"},
        ]
    }
)


def test_shipped_registry_loads():
    assert isinstance(load_registry(), ModelRegistry)


def test_resolves_model_for_provider_and_role(make_settings):
    s = make_settings(
        llm_provider="groq", groq_api_key="k", llm_model="general", llm_model_code="coder"
    )
    assert resolve_model(s, registry=REGISTRY).supports_tools is True
    assert resolve_model(s, "code", registry=REGISTRY).id == "coder"


def test_model_on_wrong_provider_is_an_error(make_settings):
    s = make_settings(llm_provider="openrouter", openrouter_api_key="k", llm_model="coder")
    with pytest.raises(ConfigError, match="not in configs/models.yaml"):
        resolve_model(s, registry=REGISTRY)


def test_missing_provider_is_an_error(make_settings):
    with pytest.raises(ConfigError, match="LLM_PROVIDER is not set"):
        resolve_model(make_settings(llm_model="general"), registry=REGISTRY)


def test_duplicate_entries_are_rejected():
    with pytest.raises(ValidationError, match="listed twice"):
        ModelRegistry.model_validate(
            {"models": [{"id": "a", "provider": "groq"}, {"id": "a", "provider": "groq"}]}
        )


def test_openai_api_models_are_rejected_but_gpt_oss_allowed():
    with pytest.raises(ValidationError, match="OpenAI API model"):
        ModelRegistry.model_validate(
            {"models": [{"id": "openai/gpt-4o", "provider": "openrouter"}]}
        )
    ModelRegistry.model_validate(
        {"models": [{"id": "openai/gpt-oss-120b", "provider": "openrouter"}]}
    )


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        ModelRegistry.model_validate({"models": [{"id": "a", "provider": "groq", "rmp": 5}]})
