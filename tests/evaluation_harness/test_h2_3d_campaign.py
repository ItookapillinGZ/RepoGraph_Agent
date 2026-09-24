"""Offline contract tests for the clean H2.3D paired campaign."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.artifact_security import ArtifactSecurityError, scan_persisted_artifacts
from evaluation.campaign import CampaignError, CampaignStopped
from evaluation.campaign_h2_3d import (
    H2_3D_BASE_TASK_RUNS,
    H2_3D_RETRY_RESERVE,
    MAX_H2_3D_LIVE_TASK_RUNS,
    H23DCampaign,
    _direct_openai_identity,
    _status_code,
)
from evaluation.models import EvaluationResult
from model_defaults import ProductionModelSettings


def settings(
    *,
    provider: str = "openai",
    model: str = "gpt-5.6-luna",
    base_url: str | None = None,
) -> ProductionModelSettings:
    return ProductionModelSettings(
        provider=provider,
        model=model,
        temperature=0.0,
        seed=None,
        api_key="configured-test-value",
        base_url=base_url,
        timeout_seconds=300.0,
        reasoning_effort="none",
        max_completion_tokens=8192,
    )


def result(
    task_id: str,
    *,
    resolved: bool = False,
    infrastructure: bool = False,
    candidate: bool = True,
) -> EvaluationResult:
    status = "resolved" if resolved else "unresolved"
    if infrastructure:
        status = "evaluation_error"
    elif not candidate:
        status = "agent_failed"
    return EvaluationResult(
        task_id=task_id,
        dataset="repograph-local-h2-3-v1",
        experiment="experiment",
        repository="repository",
        base_commit="commit",
        created_at="now",
        status=status,
        final_resolved=resolved,
        valid_prediction=candidate and not infrastructure,
        infrastructure_failure=infrastructure,
        duration_seconds=1,
    )


class H23DContractTests(unittest.TestCase):
    def test_clean_campaign_budget_is_68_plus_12(self) -> None:
        self.assertEqual(H2_3D_BASE_TASK_RUNS, 68)
        self.assertEqual(H2_3D_RETRY_RESERVE, 12)
        self.assertEqual(MAX_H2_3D_LIVE_TASK_RUNS, 80)

    def test_sdk_default_is_direct_openai(self) -> None:
        identity = _direct_openai_identity(settings())
        self.assertEqual(identity["provider_identity"], "openai")
        self.assertEqual(identity["routing"], "direct")
        self.assertEqual(identity["endpoint_identity"], "SDK default")

    def test_public_openai_host_is_direct_even_with_compatible_adapter(self) -> None:
        identity = _direct_openai_identity(
            settings(
                provider="openai_compatible",
                base_url="https://api.openai.com/v1",
            )
        )
        self.assertEqual(identity["provider_adapter_mode"], "openai_compatible")
        self.assertEqual(identity["endpoint_identity"], "https://api.openai.com")

    def test_relay_or_wrong_model_is_rejected_before_live_calls(self) -> None:
        with self.assertRaises(CampaignError):
            _direct_openai_identity(
                settings(
                    provider="openai_compatible",
                    base_url="https://relayapi.example/v1",
                )
            )
        with self.assertRaises(CampaignError):
            _direct_openai_identity(settings(model="gpt-5.6-sol"))

    def test_resolved_model_mismatch_is_a_hard_stop(self) -> None:
        H23DCampaign._validate_resolved_models(["gpt-5.6-luna"])
        H23DCampaign._validate_resolved_models([])
        with self.assertRaises(CampaignStopped):
            H23DCampaign._validate_resolved_models(["gpt-5.6-sol"])

    def test_matrix_statuses_preserve_infrastructure_and_non_candidate(self) -> None:
        self.assertEqual(_status_code(result("r", resolved=True)), "R")
        self.assertEqual(_status_code(result("u")), "U")
        self.assertEqual(_status_code(result("i", infrastructure=True)), "I")
        self.assertEqual(_status_code(result("n", candidate=False)), "N")

    def test_partial_usage_never_becomes_exact_tokens_per_task(self) -> None:
        complete = result("complete").model_copy(
            update={
                "llm_calls": 2,
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }
        )
        partial = result("partial").model_copy(
            update={"llm_calls": 1, "telemetry_incomplete": True}
        )
        summary = H23DCampaign._telemetry([complete, partial])
        self.assertEqual(summary["tasks_with_complete_usage"], 1)
        self.assertEqual(summary["tasks_with_incomplete_usage"], 1)
        self.assertEqual(summary["total_tokens"], 15)
        self.assertIsNone(summary["tokens_per_task"])
        self.assertIsNone(summary["estimated_cost_usd"])

    def test_security_scan_includes_sqlite_and_log_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluation.db").write_bytes(b"safe sqlite bytes")
            (root / "worker.log").write_text("safe log", encoding="utf-8")
            scan = scan_persisted_artifacts([root], environment={})
            self.assertEqual(scan.files_scanned, 2)
            (root / "evaluation.db").write_bytes(b"sk-" + b"x" * 20)
            with self.assertRaises(ArtifactSecurityError):
                scan_persisted_artifacts([root], environment={})


if __name__ == "__main__":
    unittest.main()
