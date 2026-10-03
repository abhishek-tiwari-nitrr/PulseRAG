"""I/P and O/P safety checks backed by Groq hosted guardrails."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass

import httpx

from pulserag.core.settings import AppSettings

logger = logging.getLogger(__name__)
__all__ = [
    "GuardrailDecision",
    "check_prompt_injection",
    "check_safeguard_policy",
]

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_RETRY_BACKOFF_SECONDS = 0.25
_CLIENT_CACHE: dict[tuple[str, float], httpx.Client] = {}
_CLIENT_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class GuardrailDecision:
    """The verdict from one check."""

    allowed: bool
    reason: str | None = None
    score: float | None = None
    latency_ms: float | None = None


def check_prompt_injection(question: str, settings: AppSettings) -> GuardrailDecision:
    """Score a question prompt injection intent and decide whether to run it."""
    client = _get_client(settings)
    if client is None:
        return GuardrailDecision(allowed=True, reason=None)

    started = time.perf_counter()
    try:
        content = _chat_completion(
            client=client,
            model=settings.groq_prompt_guard_model,
            message=[{"role": "user", "content": question}],
            max_retries=settings.groq_max_retries,
        )
        score = float(content.strip())
    except Exception:
        logger.warning(
            "Input guardrail unavailable; allowing question through.",
            extra={"guardrails": "prompt_guard"},
            exc_info=True,
        )
        return GuardrailDecision(allowed=True, latency_ms=_elapsed_ms(started))
    latency_ms = _elapsed_ms(started)
    if score < settings.prompt_guard_threshold:
        return GuardrailDecision(allowed=True, score=score, latency_ms=latency_ms)

    reason = f"Prompt Guard scored this question {score:.4f} at or above the configured threshold."
    logger.info(
        "Guaradrail blocked request at input",
        extra={"guardrail": "prompt_guard", "score": score, "latency_ms": latency_ms},
    )
    return GuardrailDecision(
        allowed=False, reason=reason, latency_ms=latency_ms, score=score
    )


def check_safeguard_policy(
    question: str, answer: str, policy: str, settings: AppSettings
) -> GuardrailDecision:
    """Judge a drafted answer against the safety policy."""
    client = _get_client(settings)
    if client is None:
        return GuardrailDecision(allowed=True, reason=None)

    started = time.perf_counter()

    try:
        content = _chat_completion(
            client=client,
            model=settings.groq_safeguard_model,
            max_retries=settings.groq_max_retries,
            message=[
                {"role": "system", "content": policy},
                {
                    "role": "user",
                    "content": f"USER_QUESTION: {question} \n\n ASSISTANT_ANSWER: {answer}",
                },
            ],
        )
        verdict = json.loads(content)
    except Exception:
        logger.warning(
            "Ouput Guardrail unavailable; allowing answer through.",
            extra={"guardrail": "safeguard"},
            exc_info=True,
        )
        return GuardrailDecision(allowed=True, latency_ms=_elapsed_ms(started))
    latency_ms = _elapsed_ms(started)
    if not verdict.get("violation"):
        return GuardrailDecision(allowed=True, latency_ms=latency_ms)

    reason = str(
        verdict.get("rationale")
        or verdict.get("category")
        or "Answer violated the saftey policy."
    )
    logger.info(
        "Guardrail blocked request at output.",
        extra={"guardrail": "safeguard", "category": verdict.get("category")},
    )
    return GuardrailDecision(allowed=False, reason=reason, latency_ms=latency_ms)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1_000, 2)


def _chat_completion(
    client: httpx.Client, model: str, message: list[dict[str, str]], max_retries: int
) -> str:
    """POST a chat completion and return the assistant message text."""
    last_error: Exception = RuntimeError("Guardrail call was never attempted.")
    for attempt in range(max_retries + 1):
        try:
            response = client.post(
                "/chat/completions", json={"model": model, "messages": message}
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return str(content)
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in _RETRYABLE_STATUS:
                raise
            last_error = e
        if attempt < max_retries:
            time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_error


def _get_client(settings: AppSettings) -> httpx.Client | None:
    if not settings.guardrails_active:
        return None
    api_key = settings.secret(settings.groq_api_key)
    if not api_key:
        return None
    key = (api_key, settings.groq_timeout_seconds)

    client = _CLIENT_CACHE.get(key)
    if client is not None:
        return client

    with _CLIENT_CACHE_LOCK:
        client = _CLIENT_CACHE.get(key)
        if client is None:
            client = httpx.Client(
                base_url=GROQ_BASE_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=settings.groq_timeout_seconds,
            )
            _CLIENT_CACHE[key] = client
        return client
