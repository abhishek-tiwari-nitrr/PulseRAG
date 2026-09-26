"""Typed, validated application configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["AppSettings", "get_settings"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class AppSettings(BaseSettings):
    """Runtime Config read from environment and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # application
    active_project: str = Field(default="pulserag")
    log_level: LogLevel = Field(default="INFO")

    # vector store
    qdrant_url: str
    qdrant_api_key: SecretStr | None = Field(default=None)
    qdrant_timeout_seconds: Annotated[float, Field(gt=0)] = Field(default=30.0)

    # generation
    openai_api_key: SecretStr | None = Field(default=None)
    openai_model: str = Field(default="gpt-5-mini")
    openai_temperature: Annotated[float, Field(ge=0.0, le=2.0)] = Field(default=0.1)
    openai_timeout_seconds: Annotated[float, Field(gt=0)] = Field(default=60.0)
    openai_max_retries: Annotated[int, Field(ge=0, le=5)] = Field(default=3)

    # embeddings
    embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI Embedding Model"
    )
    embedding_dimensions: Annotated[int, Field(gt=0)] = Field(default=512)
    embedding_batch_size: Annotated[int, Field(ge=1, le=10)] = Field(default=8)

    # guardrails
    groq_api_key: SecretStr | None = Field(default=None)

    # chunking
    # 2^7 = 128 to 2^13 = 8192
    chunk_size: Annotated[int, Field(ge=128, le=8192)] = Field(default=512)
    # 2^12 = 4096
    chunk_overlap: Annotated[int, Field(ge=0, le=4096)] = Field(default=100)

    # retrieval
    similarity_top_k: Annotated[int, Field(ge=1, le=100)] = Field(default=10)
    similarity_cutoff: Annotated[float, Field(ge=0.0, le=1.0)] | None = Field(
        default=0.5
    )

    # ingestion
    llama_cloud_api_key: SecretStr | None = Field(default=None)
    data_dir: Path | None = Field(default=None)
    max_guideline_files: Annotated[int, Field(ge=0, le=5)] = Field(default=3)
    pubmed_enabled: bool = Field(default=True)
    pubmed_query_limit: Annotated[int, Field(ge=0)] = Field(default=1)
    pubmed_max_results: Annotated[int, Field(ge=1, le=10)] = Field(default=5)
    include_bootstrap_documents: bool = Field(default=True)

    # validators
    @field_validator(
        "qdrant_api_key",
        "openai_api_key",
        "llama_cloud_api_key",
        "groq_api_key",
        mode="before",
    )
    @classmethod
    def _blank_secret_is_absent(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_chunk_overlap(self) -> AppSettings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP: {self.chunk_overlap} must be smaller than CHUNK_SIZE: {self.chunk_size}"
            )
        return self

    def secret(self, value: SecretStr | None) -> str | None:
        """Keeping keys wrapped."""
        return value.get_secret_value() if value is not None else None


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
