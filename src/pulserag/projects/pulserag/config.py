"""Declarative configuration for the clinical-guideline project."""

from __future__ import annotations

from pulserag import PACKAGE_ROOT
from pulserag.core.project import ProjectConfig
from pulserag.core.settings import AppSettings

__all__ = [
    "COLLECTION_NAME",
    "DISCLAIMER",
    "PROJECT_NAME",
    "SYSTEM_PROMPT",
    "build_pulserag_config",
]

PROJECT_NAME = "pulserag"
COLLECTION_NAME = "pulserag_collection_openai_small"
SYSTEM_PROMPT = "You are a clinical guidelines assistant. Answer medical questions using only the retrieved context. Cite source organizations when possible, state uncertainty when the context is incomplete, and do not provide personalized medical advice."
DISCLAIMER = "For educational purposes only. This is not medical advice."


def build_pulserag_config(settings: AppSettings) -> ProjectConfig:
    """Assemble the project config from settings."""
    project_package = PACKAGE_ROOT / "projects" / PROJECT_NAME
    data_dir = (
        settings.data_dir if settings.data_dir is not None else project_package / "data"
    )
    return ProjectConfig(
        name=PROJECT_NAME,
        collection_name=COLLECTION_NAME,
        system_prompt=SYSTEM_PROMPT,
        disclaimer=DISCLAIMER,
        data_dir=data_dir,
    )
