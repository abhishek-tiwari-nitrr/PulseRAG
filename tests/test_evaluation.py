"""Evaluation, with fake judges instead of DeepEval's paid LLM calls."""

import json
import os
from types import SimpleNamespace

from pulserag.core.evaluation import load_latest_eval_result, run_evaluation, runner
from pulserag.core.evaluation.metrics import build_rag_metrics
from pulserag.core.evaluation.runner import load_eval_dataset
from pulserag.core.schemas import QueryArtifacts, RAGResponse
from pulserag.projects.pulserag.config import build_pulserag_config

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")


class FakeMetric:
    """Behaves like a DeepEval metric: measure() stores score, success and reason."""

    def __init__(self, name, crash=False):
        self.name = name
        self.threshold = 0.7
        self.crash = crash

    def measure(self, test_case, _show_indicator=False):
        if self.crash:
            raise RuntimeError("judge was rate-limited")
        self.score, self.success, self.reason = 0.9, True, "Supported by the context."


def fake_service(config, settings):
    answer = RAGResponse(answer="Metformin.", evidence="...", citations=[], confidence="high",
                         disclaimer=config.disclaimer)
    return SimpleNamespace(
        config=config,
        settings=settings,
        collection_ready=lambda: True,
        query=lambda question: QueryArtifacts(response=answer, retrieval_context=["context"]),
    )


def write_dataset(config):
    config.golden_dataset_path.write_text(json.dumps([
        {"id": "a", "query": "First-line therapy?", "expected_answer": "Metformin."},
        {"id": "b", "query": "What does the PDF say?", "expected_answer": "...",
         "requires": ["guideline_pdf"]},
    ]))


def test_the_shipped_golden_dataset_is_valid(make_settings):
    cases = load_eval_dataset(build_pulserag_config(make_settings()))

    assert cases
    assert len({case["id"] for case in cases}) == len(cases)


def test_the_three_metrics_have_readable_names(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    names = [metric.name for metric in build_rag_metrics()]

    assert names == ["Answer Relevance", "Context Relevance", "Groundedness"]


def test_a_run_scores_cases_skips_unindexed_sources_and_saves_the_report(
    config, make_settings, monkeypatch
):
    write_dataset(config)
    monkeypatch.setattr(runner, "build_rag_metrics", lambda: [FakeMetric("Groundedness")])

    report = run_evaluation(fake_service(config, make_settings(max_guideline_files=0)))

    assert report.summary.success is True
    assert report.summary.passed_cases == 1
    assert report.summary.skipped_case_ids == ["b"]  # needs the PDF, which is switched off
    assert report.cases[0].metrics[0].score == 0.9
    assert load_latest_eval_result(config) == report


def test_a_crashing_metric_fails_its_case_but_not_the_run(config, make_settings, monkeypatch):
    write_dataset(config)
    monkeypatch.setattr(runner, "build_rag_metrics",
                        lambda: [FakeMetric("Groundedness", crash=True)])

    report = run_evaluation(fake_service(config, make_settings(max_guideline_files=0)))

    assert report.summary.failed_cases == 1
    assert report.cases[0].metrics[0].error == "judge was rate-limited"
