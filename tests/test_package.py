import importlib

import pytest

import srs_agent

SUBPACKAGES = [
    "llm",
    "ingest",
    "requirements",
    "memory",
    "agent",
    "testing",
    "sandbox",
    "storage",
    "api",
]


def test_version():
    assert srs_agent.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    module = importlib.import_module(f"srs_agent.{name}")
    assert module.__doc__
