"""The contract every domain plugin implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core.schema import Document

    from pulserag.core.settings import AppSettings

__all__ = ["DocumentIngestor", "ProjectConfig"]


@dataclass(frozen=True)
class ProjectConfig:
    """Everything Core Required. Forzen because these values are read on every query."""

    name: str
    collection_name: str
    system_prompt: str
    disclaimer: str
    data_dir: Path

    @property
    def guideline_dir(self):
        """Dir holding operator supplied source pdfs."""
        return self.data_dir / "guidelines"


class DocumentIngestor(ABC):
    """Raw Source into labelled LlamaIndex documents."""

    def __init__(self, config: ProjectConfig, settings: AppSettings):
        self.config = config
        self.settings = settings

    @abstractmethod
    def load_and_parse(self) -> list[Document]:
        """Load raw source and return LlamaIndex documents."""

    @abstractmethod
    def enrich_metadata(self, docs: list[Document]) -> list[Document]:
        """Attach the metadata the source citations are built from."""

    def ingest(self) -> list[Document]:
        """Load, Parse and label corpus."""
        return self.enrich_metadata(self.load_and_parse())
