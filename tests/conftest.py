"""Shared test setup. Every test runs offline: no API keys, no network, no real .env."""

import os

import pytest

from pulserag.core.project import ProjectConfig
from pulserag.core.settings import AppSettings, get_settings

_APP_VARIABLES = (
    "ACTIVE_PROJECT",
    "LOG_LEVEL",
    "QDRANT_",
    "OPENAI_",
    "EMBEDDING_",
    "CHUNK_",
    "SIMILARITY_",
    "DATA_DIR",
    "MAX_",
    "LLAMA_",
    "PUBMED_",
    "INCLUDE_",
)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    """Keep your shell's variables and the cached settings out of every test."""
    for name in list(os.environ):
        if name.startswith(_APP_VARIABLES):
            monkeypatch.delenv(name)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def make_settings():
    """Build settings from explicit values only; `_env_file=None` ignores .env."""

    def _make(**overrides):
        values = {
            "_env_file": None,
            "qdrant_url": "http://qdrant.invalid:6333",
            "openai_api_key": "sk-test",
            **overrides,
        }
        return AppSettings(**values)

    return _make


@pytest.fixture
def config(tmp_path):
    """A project config whose data folder is a temporary directory."""
    return ProjectConfig(
        name="pulserag",
        collection_name="test_collection",
        system_prompt="You are a test assistant.",
        disclaimer="Not medical advice.",
        data_dir=tmp_path,
        golden_dataset_path=tmp_path / "golden_dataset.json",
    )
