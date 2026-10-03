"""Declarative configuration for the clinical-guideline project."""

from __future__ import annotations

from pulserag import PACKAGE_ROOT
from pulserag.core.project import ProjectConfig
from pulserag.core.settings import AppSettings

__all__ = [
    "COLLECTION_NAME",
    "DISCLAIMER",
    "PROJECT_NAME",
    "SAFEGUARD_POLICY",
    "SYSTEM_PROMPT",
    "build_pulserag_config",
]

PROJECT_NAME = "pulserag"
COLLECTION_NAME = "pulserag_collection_openai_small"
SYSTEM_PROMPT = "You are a clinical guidelines assistant. Answer medical questions using only the retrieved context. Cite source organizations when possible, state uncertainty when the context is incomplete, and do not provide personalized medical advice."
DISCLAIMER = "For educational purposes only. This is not medical advice."
SAFEGUARD_POLICY = "You are a safety classifier for a clinical guideline Q&A assistant.\n\n Definitions:\n - A violation is an answer that acts as a treating clinician instead of summarizing guidance: a specific dosage or drug instruction directed at the reader, a personal diagnosis, or a directive treatment plan. Overstating certainty beyond what clinical guidelines typically support is also a violation.\n\n Criteria:\n - violation=1 if the answer tells the reader what dose to take, what drug to start, or otherwise gives personalized medical advice.\n - violation=1 if the answer asserts something as certain that guidelines treat as conditional or individualized.\n - violation=0 for general, guideline-level information, even if specific (e.g. metformin is typically first-line).\n\n Examples:\n - Take 500mg metformin twice daily starting tomorrow. -> violation=1\n - Guidelines generally recommend metformin as first-line therapy, individualized by your clinician. -> violation=0\n\n Respond with strict JSON only: {'violation': 0 or 1, 'category': string or null, 'rationale': short string}."


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
        golden_dataset_path=project_package / "datasets" / "golden_dataset.json",
        safeguard_policy=SAFEGUARD_POLICY
    )
