import pytest

from srs_agent.config import Settings

ENV_NAMES = [name.upper() for name in Settings.model_fields]


@pytest.fixture
def make_settings(monkeypatch):
    """Build Settings from the given values only, ignoring the real .env and shell."""

    def make(env_file=None, **values):
        for name in ENV_NAMES:
            monkeypatch.delenv(name, raising=False)
        for key, value in values.items():
            monkeypatch.setenv(key.upper(), value)
        return Settings(_env_file=env_file)

    return make
