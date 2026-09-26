"""Lookup table mapping a project name to its plugin implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pulserag.core.project import DocumentIngestor, ProjectConfig
from pulserag.core.settings import AppSettings


@dataclass(frozen=True)
class ProjectDefinition:
    """Plugin: configuration and configuration."""

    config: ProjectConfig
    ingestor_class: type[DocumentIngestor]


ProjectFactory = Callable[[AppSettings], ProjectDefinition]


def _build_pulserag(settings: AppSettings) -> ProjectDefinition:
    """Built the clinical guideline project definition."""
    from pulserag.projects.pulserag.config import build_pulserag_config
    from pulserag.projects.pulserag.ingestor import PulseRAGIngestor

    return ProjectDefinition(
        config=build_pulserag_config(settings=settings), ingestor_class=PulseRAGIngestor
    )


PROJECT_FACTORIES: dict[str, ProjectFactory] = {
    "pulserag": _build_pulserag,
}


def available_projects() -> list[str]:
    """Return registered project names, sorted."""
    return sorted(PROJECT_FACTORIES)


def get_project_definition(name: str, settings: AppSettings) -> ProjectDefinition:
    """Resolve a project name to its plugin definition."""
    try:
        factory = PROJECT_FACTORIES[name]
    except KeyError as e:
        supported = ", ".join(available_projects())
        raise ValueError(
            f"Unkonwn Project '{name}. Registered Projects: {supported}'"
        ) from e
    return factory(settings)
