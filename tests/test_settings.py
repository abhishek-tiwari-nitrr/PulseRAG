"""Settings and logging."""

import logging

import pytest
from pydantic import ValidationError

from pulserag.core.logging import configure_logging
from pulserag.core.settings import AppSettings


def test_defaults_and_masked_secrets(make_settings):
    settings = make_settings()

    assert settings.openai_model == "gpt-4o-mini"
    assert settings.embedding_dimensions == 512
    assert "sk-test" not in repr(settings)
    assert settings.secret(settings.openai_api_key) == "sk-test"


def test_a_blank_key_counts_as_unset(make_settings):
    settings = make_settings(openai_api_key="   ", llama_cloud_api_key="")

    assert settings.openai_api_key is None
    assert settings.llama_cloud_api_key is None


def test_invalid_configuration_fails_at_start_up(make_settings):
    with pytest.raises(ValidationError, match="qdrant_url"):
        AppSettings(_env_file=None)
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        make_settings(chunk_size=200, chunk_overlap=200)


def test_configure_logging_can_be_called_twice():
    configure_logging("DEBUG")
    configure_logging("INFO")

    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.INFO
