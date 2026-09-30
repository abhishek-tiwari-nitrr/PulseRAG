"""Answer-quality evaluation against a checked-in golden dataset."""

from __future__ import annotations

from pulserag.core.evaluation.runner import load_latest_eval_result, run_evaluation

__all__ = ["load_latest_eval_result", "run_evaluation"]