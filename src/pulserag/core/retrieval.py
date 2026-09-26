"""Assembling the query engine."""

from __future__ import annotations

import logging
from typing import Any

from pulserag.core.project import ProjectConfig
from pulserag.core.settings import AppSettings

logger = logging.getLogger(__name__)
__all__ = ["build_llm"]


def build_llm(config: ProjectConfig, settings: AppSettings) -> Any:
    """Construct the generation LLM."""
    from llama_index.llms.openai import OpenAI

    api_key = settings.secret(settings.openai_api_key)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set, so answer cannot be generated.")
    return OpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        system_prompt=config.system_prompt,
        api_key=api_key,
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )


def build_query_engine(index: Any, config: ProjectConfig, settings: AppSettings) -> Any:
    """Compose retriever, post processors and synthesizer into query engine."""
    query_kwargs: dict[str, Any] = {
        "llm": build_llm(config, settings),
        "similarity_top_k": settings.similarity_top_k,
        "response_mode": "compact",
    }

    post_processors = _build_post_processors(settings)
    if post_processors:
        query_kwargs["node_postprocessors"] = post_processors

    logger.info(
        "Built query enigne",
        extra={
            "similarity_cutoff": settings.similarity_cutoff,
            "similarity_top_k": settings.similarity_top_k,
        },
    )
    return index.as_query_engine(**query_kwargs)


def _build_post_processors(settings: AppSettings) -> list[Any]:
    """Build the node post-processor chain."""
    if settings.similarity_cutoff is None:
        return []

    from llama_index.core.postprocessor import SimilarityPostprocessor

    return [SimilarityPostprocessor(similarity_cutoff=settings.similarity_cutoff)]
