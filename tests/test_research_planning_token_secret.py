"""Focused tests for planning-token signing-key initialization."""

from __future__ import annotations

import hashlib
import hmac
import logging

from backend.services import research_planning_service


def test_configured_recon_token_secret_is_used_as_shared_key(monkeypatch):
    monkeypatch.setenv("RECON_TOKEN_SECRET", "x" * 32)

    assert research_planning_service._load_recon_token_secret() == b"x" * 32


def test_missing_recon_token_secret_generates_fresh_process_key_and_warns(monkeypatch, caplog):
    monkeypatch.delenv("RECON_TOKEN_SECRET", raising=False)
    generated = iter((b"a" * 32, b"b" * 32))
    monkeypatch.setattr(research_planning_service.secrets, "token_bytes", lambda size: next(generated))
    caplog.set_level(logging.WARNING, logger=research_planning_service.__name__)

    first = research_planning_service._load_recon_token_secret()
    second = research_planning_service._load_recon_token_secret()

    assert first == b"a" * 32
    assert second == b"b" * 32
    assert "process-local" in caplog.text
    assert "another worker" in caplog.text
    assert "restart" in caplog.text
    assert first.hex() not in caplog.text
    assert second.hex() not in caplog.text


def test_process_local_key_supports_same_process_recon_and_plan_round_trips(monkeypatch):
    monkeypatch.setattr(research_planning_service, "_RECON_TOKEN_SECRET", b"k" * 32)
    context = "Harness 是用于约束和审计 Agent 执行的运行时。"
    plan = {
        "version": "v1",
        "user_request": "调研 Harness",
        "sections": [{"title": "核心机制", "questions": [{"text": "如何工作？"}]}],
    }

    token = research_planning_service.recon_token(context)
    signed_plan = research_planning_service.plan_token(plan)

    assert research_planning_service.recon_context_from_token(token) == context
    assert research_planning_service.verify_plan_token(plan, signed_plan) is True


def test_tokens_from_another_process_key_are_rejected(monkeypatch):
    context = "跨进程凭证"
    monkeypatch.setattr(research_planning_service, "_RECON_TOKEN_SECRET", b"a" * 32)
    token = research_planning_service.recon_token(context)

    monkeypatch.setattr(research_planning_service, "_RECON_TOKEN_SECRET", b"b" * 32)

    assert research_planning_service.recon_context_from_token(token) is None


def test_removed_public_default_legacy_token_is_rejected(monkeypatch):
    context = "旧公开默认密钥签发的澄清上下文"
    old_public_token = hmac.new(
        b"harness-agent-recon-token-v1",
        context.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:32]
    monkeypatch.setattr(research_planning_service, "_RECON_TOKEN_SECRET", b"private-test-secret" * 2)

    result = research_planning_service.draft_research_plan(
        "调研旧公开默认 token",
        recon_context=context,
        recon_token_value=old_public_token,
        research_service_module=object(),
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "invalid_recon_token"
