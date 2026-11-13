from pathlib import Path

import pytest
from pydantic import ValidationError

from srs_agent.config import ConfigError, load_run_config


def test_defaults_when_nothing_set(make_settings):
    s = make_settings()
    assert s.llm_provider is None
    assert s.run_mode == "dev"
    assert s.default_project_path == Path.home() / "SRSProjects"


def test_reads_values_and_hides_keys(make_settings):
    s = make_settings(llm_provider="groq", llm_model="m1", groq_api_key="secret-123")
    assert s.groq_api_key.get_secret_value() == "secret-123"
    assert "secret-123" not in repr(s)


def test_old_variables_are_ignored(make_settings):
    s = make_settings(gemini_api_key="x", groq_model="old", groq_preferred_models="a,b")
    assert not hasattr(s, "gemini_api_key")


def test_blank_values_mean_unset(make_settings):
    s = make_settings(llm_model="", default_project_path="")
    assert s.llm_model is None
    assert s.default_project_path == Path.home() / "SRSProjects"


def test_windows_line_endings_are_stripped(make_settings, tmp_path):
    env = tmp_path / ".env"
    env.write_bytes(b"LLM_PROVIDER=groq\r\nGROQ_API_KEY=abc\r\nLLM_MODEL=m1\r\n")
    s = make_settings(env_file=env)
    assert s.groq_api_key.get_secret_value() == "abc"
    assert s.llm_model == "m1"


def test_missing_key_for_provider_is_an_error(make_settings):
    with pytest.raises(ValidationError, match="GROQ_API_KEY is not set"):
        make_settings(llm_provider="groq")


def test_unknown_provider_is_an_error(make_settings):
    with pytest.raises(ValidationError):
        make_settings(llm_provider="gemini")


def test_project_path_must_be_absolute(make_settings):
    with pytest.raises(ValidationError, match="absolute"):
        make_settings(default_project_path=r"D:\SRSProjects")


def test_no_fallback_in_experiment_mode(make_settings):
    with pytest.raises(ValidationError, match="LLM_FALLBACK"):
        make_settings(run_mode="experiment", llm_fallback="other-model")


def test_role_override_wins_over_default_model(make_settings):
    s = make_settings(llm_model="general", llm_model_code="coder")
    assert s.model_for("code") == "coder"
    assert s.model_for("plan") == "general"
    assert s.model_for() == "general"


def test_model_for_without_model_is_a_clear_error(make_settings):
    with pytest.raises(ConfigError, match="LLM_MODEL is not set"):
        make_settings().model_for("analyze")


def test_base_and_memory_configs_differ_only_in_memory():
    base, memory = load_run_config("base"), load_run_config("memory")
    assert base.memory.enabled is False
    assert memory.memory.enabled is True
    assert base.model_dump(exclude={"name", "memory"}) == memory.model_dump(
        exclude={"name", "memory"}
    )


def test_run_config_rejects_unknown_fields(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("name: x\nmemory:\n  enabeld: true\n")
    with pytest.raises(ValidationError):
        load_run_config(path)


def test_missing_run_config_is_a_clear_error():
    with pytest.raises(ConfigError, match="not found"):
        load_run_config("does-not-exist")
