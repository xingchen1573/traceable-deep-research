from __future__ import annotations

from scripts.replay_research_evidence_native_qualification import (
    ATOMIC_BOUNDARY_PROFILE,
    ATOMIC_BOUNDARY_PROMPT_VERSION,
    _generation_failures,
    _provider_qualification,
    _quality_payload_profile,
)
from backend.research.quality import (
    ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2,
    ENTAILMENT_PAYLOAD_PROFILE_DEFAULT,
    _judge_batch,
)


class _FakeRecordingModel:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []
        self.payloads: list[dict] = []
        self.kwargs: list[dict] = []

    def __call__(self, payload: dict, **kwargs: object) -> dict:
        self.payloads.append(payload)
        self.kwargs.append(kwargs)
        return self.response


def test_generation_failures_reject_model_failed_or_empty_sections() -> None:
    failures = _generation_failures({
        "documents": [
            {
                "section_id": "sec_01",
                "content": "",
                "synthesis_diagnostics": {"mode": "model_failed"},
            }
        ]
    })

    assert failures == [
        {
            "section_id": "sec_01",
            "mode": "model_failed",
            "failure_reason": "writer_returned_no_deliverable_content",
        }
    ]


def test_generation_failures_accept_deliverable_content() -> None:
    assert _generation_failures({
        "documents": [
            {
                "section_id": "sec_01",
                "content": "有证据支持的正文[1]。",
                "synthesis_diagnostics": {"mode": "evidence_native_writer"},
            }
        ]
    }) == []


def test_provider_qualification_requires_expected_model_without_fallback() -> None:
    passed = _provider_qualification(
        [
            {
                "status": "success",
                "response": {
                    "_experiment_model_metadata": {
                        "provider": "preferred",
                        "model": "mimo-v2.5",
                        "fallback": False,
                    }
                },
            }
        ],
        expected_model="mimo-v2.5",
    )
    mismatched = _provider_qualification(
        [
            {
                "status": "success",
                "response": {
                    "_experiment_model_metadata": {
                        "provider": "mimo_fallback",
                        "model": "mimo-v2.5",
                        "fallback": True,
                    }
                },
            }
        ],
        expected_model="mimo-v2.5",
    )

    assert passed["status"] == "passed"
    assert mismatched["status"] == "provider_mismatch"


def test_atomic_boundary_profile_scopes_evidence_per_claim_and_normalizes_result() -> None:
    inner = _FakeRecordingModel({
        "judgments": [{
            "case_id": "7",
            "status": "partial_unsupported",
            "claim_scope": "domain_fact",
            "unsupported_parts": ["额外机制"],
            "reason": "Evidence 未支持额外机制。",
        }]
    })
    result = _judge_batch(
        "sec_01",
        [(7, {
            "text": "对象采用额外机制并提高速度。",
            "claim_context": "",
            "claim_type": "inference",
            "support_scope": "domain_specific",
            "evidence_ids": ["ev_1"],
        })],
        [],
        {
            "ev_1": {"id": "ev_1", "claim": "对象提高了速度。"},
            "ev_other": {"id": "ev_other", "claim": "对象采用额外机制。"},
        },
        inner,
        ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2,
    )

    strict = inner.payloads[0]
    assert strict["prompt_version"] == ATOMIC_BOUNDARY_PROMPT_VERSION
    assert [item["id"] for item in strict["cases"][0]["evidence"]] == ["ev_1"]
    assert result[7]["claim_index"] == "7"
    assert result[7]["status"] == "unsupported"


def test_qualification_profile_maps_to_shared_production_profile() -> None:
    assert _quality_payload_profile(ATOMIC_BOUNDARY_PROFILE) == ENTAILMENT_PAYLOAD_PROFILE_ATOMIC_BOUNDARY_V2
    assert _quality_payload_profile("current") == ENTAILMENT_PAYLOAD_PROFILE_DEFAULT
