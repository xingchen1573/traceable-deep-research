from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.research import audit, model_runtime
from backend.research import replay as replay_module
from backend.research.replay import replay_model_call
from backend.services import job_service as job_service_module, research_service


class FakeChatOpenAI:
    handler = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, messages, **kwargs):
        handler = type(self).handler
        if handler is None:
            return _response('{"ok": true}')
        return handler(self, messages, kwargs)


def _response(content: str, *, model: str = "fake-model"):
    return SimpleNamespace(
        content=content,
        response_metadata={"model_name": model, "finish_reason": "stop", "token_usage": {"total_tokens": 9}},
        usage_metadata={"input_tokens": 4, "output_tokens": 5, "total_tokens": 9},
    )


@pytest.fixture(autouse=True)
def isolated_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "RESEARCH_ARTIFACT_DIR", tmp_path / "research_artifacts")
    monkeypatch.setattr(audit, "_CODE_VERSION", {"git_commit": "test", "research_source_hash": "test"})
    monkeypatch.setattr(model_runtime, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(model_runtime, "_POOL", {})
    monkeypatch.setattr(model_runtime, "_POOL_SIGNATURE", {})
    monkeypatch.setattr(model_runtime, "_PREFERRED_PROVIDER_POOL", None)
    monkeypatch.setattr(model_runtime, "_PREFERRED_PROVIDER_SIGNATURE", None)
    monkeypatch.setattr(model_runtime, "_MIMO_FALLBACK_POOL", None)
    monkeypatch.setattr(model_runtime, "_MIMO_FALLBACK_SIGNATURE", None)
    monkeypatch.setenv("RESEARCH_EVIDENCE_API_KEY", "secret-primary-key")
    monkeypatch.setenv("RESEARCH_EVIDENCE_BASE_URL", "https://primary.example/v1")
    monkeypatch.setenv("RESEARCH_MODEL_RATE_LIMIT_RETRIES", "0")
    monkeypatch.setenv("RESEARCH_MODEL_MAX_RETRIES", "0")
    monkeypatch.delenv("MIMO_FALLBACK_ENABLED", raising=False)
    monkeypatch.delenv("RESEARCH_UNIFIED_MODEL_CHANNEL_ENABLED", raising=False)
    monkeypatch.delenv("RESEARCH_PREFERRED_PROVIDER_ENABLED", raising=False)
    monkeypatch.delenv("RESEARCH_PREFERRED_GATEWAY_MANAGED", raising=False)
    monkeypatch.delenv("RESEARCH_PREFERRED_GATEWAY_RETRIES", raising=False)
    FakeChatOpenAI.handler = None
    yield
    FakeChatOpenAI.handler = None


def _enable(monkeypatch):
    monkeypatch.setenv("RESEARCH_AUDIT_ENABLED", "1")
    monkeypatch.setenv("RESEARCH_MODEL_TRACE_ENABLED", "1")


def _model_events(job_id: str) -> list[dict]:
    return [item for item in audit.read_trace_events(job_id) if item.get("event") == "model_call"]


def test_trace_disabled_preserves_call_and_writes_nothing(monkeypatch):
    monkeypatch.setenv("RESEARCH_AUDIT_ENABLED", "0")
    with audit.audit_run(job_id="job_disabled"):
        response = model_runtime.invoke_research_chat(
            [{"role": "user", "content": "hello"}], timeout_seconds=5, evidence=True
        )
    assert response.content == '{"ok": true}'
    assert not audit.trace_path("job_disabled").exists()


def test_trace_records_complete_call_and_redacts_secrets(monkeypatch):
    _enable(monkeypatch)
    with audit.audit_run(job_id="job_trace", parent_job_id="job_parent", run_id="run_test"):
        response = model_runtime.invoke_research_chat(
            [{"role": "user", "content": "Authorization: Bearer token-value api_key=secret-primary-key"}],
            timeout_seconds=5,
            evidence=True,
            system_prompt="system secret=secret-primary-key",
        )
    events = _model_events("job_trace")
    assert len(events) == 1
    event = events[0]
    assert event["job_id"] == "job_trace"
    assert event["parent_job_id"] == "job_parent"
    assert event["requested_provider"] == "primary:evidence"
    assert event["requested_model"]
    assert event["raw_response"] == '{"ok": true}'
    assert event["usage"]["total_tokens"] == 9
    assert event["input_hash"] and event["output_hash"] and event["prompt_hash"]
    raw_trace = audit.trace_path("job_trace").read_text(encoding="utf-8")
    assert "secret-primary-key" not in raw_trace
    assert "token-value" not in raw_trace
    assert "[REDACTED]" in raw_trace
    assert response.response_metadata["harness_call_id"] == event["call_id"]


def test_trace_preparation_failure_is_fail_open(monkeypatch):
    _enable(monkeypatch)
    invoked = {"value": False}

    def invoke():
        invoked["value"] = True
        return _response('{"ok": true}')

    monkeypatch.setattr(audit, "_locked_sequence", lambda _job_id: (_ for _ in ()).throw(OSError("disk")))
    with audit.audit_run(job_id="job_trace_prep_failure"):
        response = audit.invoke_with_model_trace(
            invoke,
            messages=[{"role": "user", "content": "hello"}],
            requested_provider="test",
            requested_model="test-model",
        )
    assert invoked["value"] is True
    assert response.content == '{"ok": true}'


def test_trace_classifies_and_captures_invalid_provider_response(monkeypatch):
    _enable(monkeypatch)
    provider_response = SimpleNamespace(
        status_code=200,
        headers={"content-type": "text/plain"},
        text="data: malformed api_key=secret-primary-key",
    )

    def invoke():
        error = AttributeError("'str' object has no attribute 'model_dump'")
        error.response = provider_response
        raise error

    with audit.audit_run(job_id="job_invalid_provider_response"):
        with pytest.raises(AttributeError, match="model_dump"):
            audit.invoke_with_model_trace(
                invoke,
                messages=[{"role": "user", "content": "hello"}],
                requested_provider="preferred",
                requested_model="glm-5.3-flash",
            )

    event = _model_events("job_invalid_provider_response")[0]
    assert event["status"] == "invalid_provider_response"
    assert event["error_response"] == {
        "response_type": "SimpleNamespace",
        "status_code": 200,
        "content_type": "text/plain",
        "body_excerpt": "data: malformed api_key=[REDACTED]",
    }


def test_trace_postprocessing_failure_is_fail_open(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(audit, "_response_details", lambda _response: (_ for _ in ()).throw(OSError("disk")))
    with audit.audit_run(job_id="job_trace_response_failure"):
        response = audit.invoke_with_model_trace(
            lambda: _response('{"ok": true}'),
            messages=[{"role": "user", "content": "hello"}],
            requested_provider="test",
            requested_model="test-model",
        )
    assert response.content == '{"ok": true}'


def test_timeout_and_retry_are_independent_linked_calls(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setenv("RESEARCH_MODEL_RATE_LIMIT_RETRIES", "1")
    attempts = {"count": 0}

    def handler(_model, _messages, _kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("429 rate limit")
        return _response('{"ok": true}')

    FakeChatOpenAI.handler = handler
    with audit.audit_run(job_id="job_retry"):
        model_runtime.invoke_research_chat([{"role": "user", "content": "retry"}], timeout_seconds=5)
    events = _model_events("job_retry")
    assert [event["status"] for event in events] == ["provider_error", "success"]
    assert events[1]["call_kind"] == "retry"
    assert events[1]["parent_call_id"] == events[0]["call_id"]


def test_provider_fallback_is_a_separate_child_call(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setenv("MIMO_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("MIMO_API_KEY", "secret-fallback-key")
    monkeypatch.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")

    def handler(model, _messages, _kwargs):
        if "xiaomimimo" in str(model.kwargs.get("base_url") or ""):
            return _response('{"fallback": true}', model="mimo-test")
        raise TimeoutError("primary timed out")

    FakeChatOpenAI.handler = handler
    with audit.audit_run(job_id="job_fallback"):
        response = model_runtime.invoke_research_chat(
            [{"role": "user", "content": "fallback"}],
            timeout_seconds=5,
            allow_provider_fallback=True,
        )
    events = _model_events("job_fallback")
    assert len(events) == 2
    assert events[0]["status"] == "timeout"
    assert events[1]["actual_provider"] == "mimo_fallback"
    assert events[1]["call_kind"] == "fallback"
    assert events[1]["parent_call_id"] == events[0]["call_id"]
    assert response.content == '{"fallback": true}'


def test_preferred_provider_runs_before_existing_primary(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setenv("RESEARCH_PREFERRED_PROVIDER_ENABLED", "1")
    monkeypatch.setenv("RESEARCH_PREFERRED_API_KEY", "secret-preferred-key")
    monkeypatch.setenv("RESEARCH_PREFERRED_BASE_URL", "https://preferred.example/v1")
    monkeypatch.setenv("RESEARCH_PREFERRED_MODEL_NAME", "mimo-v2.5")

    def handler(model, _messages, _kwargs):
        assert "preferred.example" in str(model.kwargs.get("base_url") or "")
        return _response('{"preferred": true}', model="mimo-v2.5")

    FakeChatOpenAI.handler = handler
    with audit.audit_run(job_id="job_preferred"):
        response = model_runtime.invoke_research_chat(
            [{"role": "user", "content": "preferred"}],
            timeout_seconds=5,
            evidence=True,
        )

    events = _model_events("job_preferred")
    assert len(events) == 1
    assert events[0]["actual_provider"] == "preferred"
    assert response.response_metadata["harness_provider"] == "preferred"


def test_preferred_provider_failure_returns_to_existing_primary(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setenv("RESEARCH_PREFERRED_PROVIDER_ENABLED", "1")
    monkeypatch.setenv("RESEARCH_PREFERRED_API_KEY", "secret-preferred-key")
    monkeypatch.setenv("RESEARCH_PREFERRED_BASE_URL", "https://preferred.example/v1")
    monkeypatch.setenv("RESEARCH_MODEL_RATE_LIMIT_RETRIES", "0")
    calls: list[str] = []

    def handler(model, _messages, _kwargs):
        base_url = str(model.kwargs.get("base_url") or "")
        calls.append(base_url)
        if "preferred.example" in base_url:
            raise TimeoutError("preferred timed out")
        return _response('{"primary": true}')

    FakeChatOpenAI.handler = handler
    with audit.audit_run(job_id="job_preferred_to_primary"):
        response = model_runtime.invoke_research_chat(
            [{"role": "user", "content": "fallback to primary"}],
            timeout_seconds=5,
            evidence=True,
        )

    events = _model_events("job_preferred_to_primary")
    assert [event["actual_provider"] for event in events] == ["preferred", "primary:evidence"]
    assert calls == ["https://preferred.example/v1", "https://primary.example/v1"]
    assert response.response_metadata["harness_provider"] == "primary"


def test_json_parse_failure_and_repair_have_separate_calls(monkeypatch):
    _enable(monkeypatch)
    responses = iter(["not json", '{"value": 3}'])

    def handler(_model, _messages, _kwargs):
        return _response(next(responses))

    FakeChatOpenAI.handler = handler
    with audit.audit_run(job_id="job_repair"):
        result = research_service._invoke_research_json(
            {"task": "Return JSON only", "schema": {"value": 0}}, timeout_seconds=5
        )
    assert result == {"value": 3}
    events = _model_events("job_repair")
    assert len(events) == 2
    assert events[1]["call_kind"] == "json_repair"
    assert events[1]["parent_call_id"] == events[0]["call_id"]
    parse_events = [item for item in audit.read_trace_events("job_repair") if item.get("event") == "model_parse"]
    assert {item["status"] for item in parse_events} >= {"parse_error", "success"}


def test_empty_response_is_recorded(monkeypatch):
    _enable(monkeypatch)
    FakeChatOpenAI.handler = lambda *_args: _response("")
    with audit.audit_run(job_id="job_empty"):
        result = research_service._invoke_research_json({"task": "empty"}, timeout_seconds=5)
    assert result["_failure_reason"] == "empty_response"
    assert _model_events("job_empty")[0]["status"] == "empty_response"


def test_thread_pool_context_does_not_cross_jobs_or_corrupt_jsonl(monkeypatch):
    _enable(monkeypatch)

    def emit_job(job_id: str):
        with audit.audit_run(job_id=job_id):
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    audit.submit_with_audit_context(
                        executor,
                        audit.invoke_with_model_trace,
                        lambda index=index: _response(json.dumps({"index": index})),
                        messages=[{"role": "user", "content": f"call {index}"}],
                        requested_provider="test",
                        requested_model="test-model",
                    )
                    for index in range(20)
                ]
                for future in futures:
                    future.result()

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(emit_job, ["job_a", "job_b"]))
    for job_id in ("job_a", "job_b"):
        events = _model_events(job_id)
        assert len(events) == 20
        assert {event["job_id"] for event in events} == {job_id}
        for line in audit.trace_path(job_id).read_text(encoding="utf-8").splitlines():
            json.loads(line)


def test_checkpoint_is_versioned_content_addressed_and_fail_open(monkeypatch):
    _enable(monkeypatch)
    with audit.audit_run(job_id="job_checkpoint", parent_job_id="job_parent", attempt=2):
        first = audit.write_stage_checkpoint(
            "planner", inputs={"request": {"q": "x"}}, outputs={"plan": {"sections": [1]}}
        )
        second = audit.write_stage_checkpoint(
            "coverage", inputs={"plan": {"sections": [1]}}, outputs={"coverage": {"complete": True}}
        )
    first_doc = json.loads(Path(first).read_text(encoding="utf-8"))
    second_doc = json.loads(Path(second).read_text(encoding="utf-8"))
    assert first_doc["schema_version"] == audit.SCHEMA_VERSION
    assert first_doc["input_artifacts"]["request"]["sha256"]
    assert second_doc["lineage"]["previous_checkpoint_id"] == first_doc["checkpoint_id"]
    monkeypatch.setattr(audit, "_atomic_write_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk")))
    with audit.audit_run(job_id="job_checkpoint_failure"):
        assert audit.write_stage_checkpoint("planner", outputs={"x": 1}) == ""


def test_checkpoint_and_model_call_have_bidirectional_lineage(monkeypatch):
    _enable(monkeypatch)
    with audit.audit_run(job_id="job_lineage"):
        request_path = audit.write_stage_checkpoint("request_normalization", outputs={"request": "x"})
        with audit.audit_scope(stage="planner"):
            response = audit.invoke_with_model_trace(
                lambda: _response('{"sections": []}'),
                messages=[{"role": "user", "content": "plan"}],
                requested_provider="test",
                requested_model="test-model",
            )
        planner_path = audit.write_stage_checkpoint("planner", outputs={"plan": {"sections": []}})

    request_doc = json.loads(Path(request_path).read_text(encoding="utf-8"))
    planner_doc = json.loads(Path(planner_path).read_text(encoding="utf-8"))
    call = _model_events("job_lineage")[0]
    call_id = response.response_metadata["harness_call_id"]
    assert call["lineage"]["input_checkpoint_id"] == request_doc["checkpoint_id"]
    assert planner_doc["lineage"]["model_calls"][0]["call_id"] == call_id
    assert planner_doc["prompt_hashes"][call_id] == call["prompt_hash"]


def test_payload_stage_prefers_coverage_over_research_question(monkeypatch):
    _enable(monkeypatch)
    payload = {"task": "Judge semantic coverage for one research question. Return JSON only."}
    assert audit.infer_stage_from_payload(payload) == "coverage_judge"
    with audit.audit_run(job_id="job_stage"):
        with audit.audit_scope(stage="coverage_judge", question_id="q1"):
            with audit.payload_audit_scope(payload):
                audit.invoke_with_model_trace(
                    lambda: _response('{"sufficient": true}'),
                    messages=[{"role": "user", "content": json.dumps(payload)}],
                    requested_provider="test",
                    requested_model="test-model",
                )
    call = _model_events("job_stage")[0]
    assert call["stage"] == "coverage_judge"
    assert call["question_id"] == "q1"


def test_parse_event_and_checkpoint_inherit_call_scope(monkeypatch):
    _enable(monkeypatch)
    with audit.audit_run(job_id="job_coverage_lineage"):
        request_path = audit.write_stage_checkpoint("request_normalization", outputs={"request": "x"})
        with audit.audit_scope(stage="coverage_judge", question_id="q1", section_id="s1"):
            response = audit.invoke_with_model_trace(
                lambda: _response('{"sufficient": true}'),
                messages=[{"role": "user", "content": "coverage"}],
                requested_provider="test",
                requested_model="test-model",
            )
        call_id = response.response_metadata["harness_call_id"]
        with audit._CALL_CONTEXT_LOCK:
            audit._CALL_CONTEXT.clear()
        audit.record_parse_result(call_id, parsed_result={"sufficient": True}, status="success")
        coverage_path = audit.write_stage_checkpoint("coverage", outputs={"complete": True})

    request_doc = json.loads(Path(request_path).read_text(encoding="utf-8"))
    coverage_doc = json.loads(Path(coverage_path).read_text(encoding="utf-8"))
    events = audit.read_trace_events("job_coverage_lineage")
    call = next(item for item in events if item.get("event") == "model_call")
    parse = next(item for item in events if item.get("event") == "model_parse")
    assert call["lineage"]["input_checkpoint_id"] == request_doc["checkpoint_id"]
    assert (parse["stage"], parse["question_id"], parse["section_id"]) == (
        call["stage"], call["question_id"], call["section_id"]
    )
    assert coverage_doc["lineage"]["model_calls"][0]["call_id"] == call_id
    assert coverage_doc["prompt_hashes"][call_id] == call["prompt_hash"]


def test_research_cancel_records_interrupt_checkpoint(monkeypatch):
    _enable(monkeypatch)
    service = job_service_module.JobService()
    source_job = {
        "job_id": "job_cancelled",
        "job_type": "research",
        "status": "running",
        "input": {"parent_job_id": "job_parent"},
        "result": {},
        "attempt": 1,
        "retry_of_job_id": None,
    }
    monkeypatch.setattr(service, "_update", lambda _job_id, mutate: mutate(dict(source_job)))
    monkeypatch.setattr(job_service_module, "emit_event", lambda *_args, **_kwargs: None)
    result = service.cancel("job_cancelled", "stop now")
    assert result["status"] == "cancelled"
    checkpoints = sorted((audit.RESEARCH_ARTIFACT_DIR / "job_cancelled" / "audit" / "checkpoints").glob("*.json"))
    assert len(checkpoints) == 1
    document = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert document["stage"] == "interrupt"
    assert document["status"] == "cancelled"
    assert document["interrupt"]["reason"] == "stop now"


def test_replay_writes_only_isolated_results(monkeypatch, tmp_path):
    _enable(monkeypatch)
    with audit.audit_run(job_id="job_source"):
        audit.invoke_with_model_trace(
            lambda: _response('{"original": true}'),
            messages=[{"role": "user", "content": "original"}],
            requested_provider="primary:evidence",
            requested_model="original-model",
            parameters={"temperature": 0, "timeout_seconds": 5, "base_url": "https://example.invalid/v1"},
        )
    original_trace = audit.trace_path("job_source").read_bytes()
    forbidden_calls = []

    def fake_invoke(call, messages, overrides):
        forbidden_calls.append((call["call_id"], messages, overrides))
        return _response('{"replayed": true}', model="ab-model")

    result = replay_model_call(
        job_id="job_source",
        stage="research_model",
        model_override="ab-model",
        user_prompt_override="experiment",
        output_root=tmp_path / "replays",
        invoke_fn=fake_invoke,
    )
    assert result["status"] == "success"
    assert result["parsed_result"] == {"replayed": True}
    assert Path(result["request_path"]).exists()
    assert Path(result["result_path"]).exists()
    request = json.loads(Path(result["request_path"]).read_text(encoding="utf-8"))
    assert request["original_call_id"] == forbidden_calls[0][0]
    assert request["side_effect_policy"]["database_write"] is False
    assert request["side_effect_policy"]["knowledge_base_write"] is False
    assert audit.trace_path("job_source").read_bytes() == original_trace


def test_replay_uses_production_parser_for_fenced_json(monkeypatch, tmp_path):
    _enable(monkeypatch)
    with audit.audit_run(job_id="job_fenced_source"):
        audit.invoke_with_model_trace(
            lambda: _response('{"original": true}'),
            messages=[{"role": "user", "content": "original"}],
            requested_provider="primary:evidence",
            requested_model="original-model",
        )
    result = replay_model_call(
        job_id="job_fenced_source",
        stage="research_model",
        output_root=tmp_path / "replays",
        invoke_fn=lambda *_args: _response('```json\n{"replayed": true}\n```'),
    )
    assert result["parsed_result"] == {"replayed": True}
    assert result["json_parse"] == {
        "parser": "backend.services.research_service._parse_json_or_error",
        "status": "success",
        "repair_mode": "none",
        "repair_attempted": False,
        "parse_error": None,
        "repair_error": None,
    }


def test_replay_records_explicit_production_json_repair(monkeypatch, tmp_path):
    _enable(monkeypatch)
    with audit.audit_run(job_id="job_repair_source"):
        audit.invoke_with_model_trace(
            lambda: _response('{"original": true}'),
            messages=[{"role": "user", "content": '{"schema": {"value": 0}}'}],
            requested_provider="primary:evidence",
            requested_model="original-model",
        )
    repair_calls = []

    def fake_repair(call, messages, content, error, overrides):
        repair_calls.append((call, messages, content, error, overrides))
        return {"value": 2}

    monkeypatch.setattr(replay_module, "_production_repair_json", fake_repair)
    result = replay_model_call(
        job_id="job_repair_source",
        stage="research_model",
        json_repair_mode="production",
        output_root=tmp_path / "replays",
        invoke_fn=lambda *_args: _response("not json"),
    )
    assert result["parsed_result"] == {"value": 2}
    assert result["json_parse"]["status"] == "repaired"
    assert result["json_parse"]["repair_attempted"] is True
    assert result["experimental_variables"]["json_repair_mode"] == "production"
    request = json.loads(Path(result["request_path"]).read_text(encoding="utf-8"))
    assert request["experimental_variables"]["json_repair_mode"] == "production"
    assert repair_calls[0][2] == "not json"


def test_replay_restores_recorded_credential_index(monkeypatch):
    monkeypatch.setattr(replay_module, "configured_keys", lambda _role: ["key-0", "key-1", "key-2"])
    monkeypatch.setattr(replay_module, "configured_mimo_fallback_keys", lambda: ["mimo-0", "mimo-1"])
    monkeypatch.setattr(replay_module, "configured_preferred_provider_keys", lambda: ["preferred-0", "preferred-1"])
    assert replay_module._credential_for_call({
        "actual_provider": "primary:evidence",
        "parameters": {"credential_index": 2},
    }) == "key-2"
    assert replay_module._credential_for_call({
        "actual_provider": "mimo_fallback",
        "parameters": {"credential_index": 1},
    }) == "mimo-1"
    assert replay_module._credential_for_call({
        "actual_provider": "preferred",
        "parameters": {"credential_index": 1},
    }) == "preferred-1"
