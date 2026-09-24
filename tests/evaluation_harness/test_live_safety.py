"""Live confirmation, no-network default, and persistence redaction tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evaluation.cli import main
from evaluation.live import LiveGuardError, run_live_preflight
from evaluation.models import EvaluationConfig
from evaluation.reports import write_reports
from evaluation.runner import create_experiment, run_evaluation_task
from evaluation.security import redact_environment_secrets
from evaluation.storage import EvaluationStorage
from model_defaults import ProductionModelSettings
from tests.evaluation_harness.helpers import make_repository, task


class RaisingAdapter:
    def __init__(self, message: str) -> None:
        self.message = message

    def run(self, *_args: object, **_kwargs: object):
        raise RuntimeError(self.message)


class LiveSafetyTests(unittest.TestCase):
    def test_run_without_live_flag_cannot_reach_preflight(self) -> None:
        preflight = MagicMock()
        with (
            patch("evaluation.cli.run_live_preflight", preflight),
            self.assertRaises(LiveGuardError),
        ):
            main(
                [
                    "run",
                    "--dataset",
                    "does-not-exist",
                    "--workspace",
                    "unused",
                    "--name",
                    "blocked",
                ]
            )
        preflight.assert_not_called()

    def test_live_preflight_requires_key_before_model_initialization(self) -> None:
        factory = MagicMock()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("evaluation.live.load_dotenv"),
            self.assertRaises(LiveGuardError),
        ):
            run_live_preflight(model_factory=factory)
        factory.assert_not_called()

    def test_provider_masked_key_fragment_is_redacted(self) -> None:
        message = "Incorrect API key provided: sk-example****************tail."
        sanitized = redact_environment_secrets(message, environment={})
        self.assertEqual(sanitized, "Incorrect API key provided: <redacted>.")

    def test_generic_provider_masked_key_fragment_is_redacted(self) -> None:
        message = "Incorrect API key provided: relay_abc********xyz."
        sanitized = redact_environment_secrets(message, environment={})
        self.assertEqual(sanitized, "Incorrect API key provided: <redacted>.")

    def test_preflight_failure_does_not_expose_provider_response(self) -> None:
        settings = ProductionModelSettings(
            provider="openai_compatible",
            model="test-model",
            temperature=0.0,
            seed=None,
            api_key="relay-secret",
            base_url="https://example.test/v1",
            timeout_seconds=30.0,
            reasoning_effort=None,
            max_completion_tokens=128,
        )

        def failing_factory():
            raise RuntimeError("provider echoed relay_abc********xyz")

        with self.assertRaisesRegex(
            LiveGuardError, r"Live API preflight failed \(RuntimeError\)\."
        ) as captured:
            run_live_preflight(model_factory=failing_factory, settings=settings)
        self.assertNotIn("relay_abc", str(captured.exception))

    def test_all_supported_provider_key_names_are_redacted(self) -> None:
        environment = {
            "CODE_REVIEW_LLM_API_KEY": "project-secret",
            "PDE_FRONTIER_LLM_API_KEY": "legacy-secret",
        }
        sanitized = redact_environment_secrets(
            "project-secret legacy-secret", environment
        )
        self.assertEqual(sanitized, "<redacted> <redacted>")

    def test_secret_is_redacted_from_result_storage_and_reports(self) -> None:
        secret = "sk-test-secret-value"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository, commit = make_repository(root)
            config = EvaluationConfig(name="redaction", agentic_test=False)
            with patch.dict("os.environ", {"OPENAI_API_KEY": secret}):
                result = run_evaluation_task(
                    task(repository, commit),
                    config,
                    workspace_root=str(root / "workspaces"),
                    experiment_id="redaction-experiment",
                    adapter=RaisingAdapter(f"provider failed with {secret}"),
                )
                experiment = create_experiment("redaction", "local", config)
                result = result.model_copy(update={"experiment": experiment.id})
                storage = EvaluationStorage(root / "evaluation.db", root / "results")
                storage.save_experiment(experiment)
                storage.save_result(result)
                markdown, structured = write_reports(
                    experiment, [result], root / "reports"
                )
            persisted = "\n".join(
                [
                    result.model_dump_json(),
                    (root / "evaluation.db")
                    .read_bytes()
                    .decode("utf-8", errors="ignore"),
                    next((root / "results").iterdir()).read_text(encoding="utf-8"),
                    markdown.read_text(encoding="utf-8"),
                    structured.read_text(encoding="utf-8"),
                ]
            )
            self.assertNotIn(secret, persisted)
            self.assertIn("<redacted>", persisted)


if __name__ == "__main__":
    unittest.main()
