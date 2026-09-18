"""Experiment-only retry and checkpoint helpers for architecture evaluation."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .finding_architecture_eval import EvaluationContractError, atomic_write_json, stable_hash


DEFAULT_504_BACKOFF_SECONDS = (5, 30, 60, 90, 120)


_HTTP_504_PATTERN = re.compile(r"(?<!\d)504(?!\d)|gateway time(?:out|-out)", re.IGNORECASE)
_TRACE_STATUS_FIELDS = ("status_code", "http_status", "response_status", "code")


def _is_http_504_value(value: Any) -> bool:
    if isinstance(value, int):
        return value == 504
    if isinstance(value, str):
        return bool(_HTTP_504_PATTERN.search(value))
    return False


def result_has_http_504(result: dict[str, Any]) -> bool:
    if not result.get("_failure_reason"):
        return False
    if _is_http_504_value(result.get("_failure_reason")):
        return True
    metadata = result.get("_experiment_model_metadata")
    if not isinstance(metadata, dict):
        return False
    for field in _TRACE_STATUS_FIELDS:
        if _is_http_504_value(metadata.get(field)):
            return True
    for event in metadata.get("trace_events") or []:
        if not isinstance(event, dict):
            continue
        for field in _TRACE_STATUS_FIELDS:
            if _is_http_504_value(event.get(field)):
                return True
        error = event.get("error")
        if isinstance(error, dict):
            for field in ("type", "message", *_TRACE_STATUS_FIELDS):
                if _is_http_504_value(error.get(field)):
                    return True
        elif _is_http_504_value(error):
            return True
    return False


class Backoff504JsonInvoker:
    """Retry only audited HTTP 504 failures, preserving every underlying trace."""

    def __init__(
        self,
        invoke: Callable[..., dict[str, Any]],
        *,
        backoff_seconds: Iterable[float] = DEFAULT_504_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.invoke = invoke
        self.backoff_seconds = tuple(max(0.0, float(value)) for value in backoff_seconds)
        self.sleep = sleep

    def __call__(
        self,
        payload: dict[str, Any],
        timeout_override: float | None = None,
    ) -> dict[str, Any]:
        attempts = []
        result: dict[str, Any] = {}
        for attempt in range(len(self.backoff_seconds) + 1):
            if timeout_override is None:
                result = self.invoke(payload)
            else:
                result = self.invoke(payload, timeout_override=timeout_override)
            is_504 = result_has_http_504(result)
            attempts.append({
                "attempt": attempt + 1,
                "http_504": is_504,
                "failure_reason": result.get("_failure_reason"),
                "response_hash": stable_hash(result),
            })
            if not is_504 or attempt >= len(self.backoff_seconds):
                break
            delay = self.backoff_seconds[attempt]
            attempts[-1]["backoff_seconds_before_next_attempt"] = delay
            self.sleep(delay)
        output = dict(result or {})
        output["_experiment_504_retry"] = {
            "schedule_seconds": list(self.backoff_seconds),
            "attempts": attempts,
            "retry_count": max(0, len(attempts) - 1),
            "exhausted": bool(attempts and attempts[-1]["http_504"]),
        }
        return output


def checkpoint_contract_hash(value: dict[str, Any]) -> str:
    return stable_hash(_normalize_contract_numbers(value))


def _normalize_contract_numbers(value: Any) -> Any:
    """Keep semantically identical CLI numbers from changing a checkpoint contract."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _normalize_contract_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_contract_numbers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_contract_numbers(item) for item in value)
    return value


def save_stage_checkpoint(
    path: Path,
    *,
    experiment_id: str,
    candidate_id: str,
    pair_id: str,
    arm: str,
    stage: str,
    source_run_sha256: str,
    contract_sha256: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    document = {
        "schema_version": "finding-architecture-experiment-stage-checkpoint/v1",
        "experiment_id": experiment_id,
        "candidate_id": candidate_id,
        "pair_id": pair_id,
        "arm": arm,
        "stage": stage,
        "source_run_sha256": source_run_sha256,
        "contract_sha256": contract_sha256,
        "payload": payload,
    }
    document["checkpoint_sha256"] = stable_hash(document)
    atomic_write_json(path, document)
    return document


def load_stage_checkpoint(
    path: Path,
    *,
    experiment_id: str,
    candidate_id: str,
    pair_id: str,
    arm: str,
    stage: str,
    source_run_sha256: str,
    contract_sha256: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = str(document.get("checkpoint_sha256") or "")
    unhashed = {key: value for key, value in document.items() if key != "checkpoint_sha256"}
    if not expected_hash or stable_hash(unhashed) != expected_hash:
        raise EvaluationContractError(f"experiment checkpoint hash mismatch: {path}")
    expected = {
        "experiment_id": experiment_id,
        "candidate_id": candidate_id,
        "pair_id": pair_id,
        "arm": arm,
        "stage": stage,
        "source_run_sha256": source_run_sha256,
        "contract_sha256": contract_sha256,
    }
    mismatches = [key for key, value in expected.items() if document.get(key) != value]
    if mismatches:
        raise EvaluationContractError(
            f"experiment checkpoint contract mismatch: {path}: {', '.join(mismatches)}"
        )
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise EvaluationContractError(f"experiment checkpoint payload invalid: {path}")
    return payload
