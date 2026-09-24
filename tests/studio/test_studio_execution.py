from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from change_set import ChangeSetReview
from plan_execution import PlanExecutionResult, PlanExecutionVerification
from studio_backend.execution import StudioExecutionAdapter
from studio_backend.models import StartRunRequest, StudioRun
from studio_backend.repositories import WorkspaceRepositories
from studio_backend.storage import StudioStorage, utc_now
from tests.studio.helpers import DIGEST, application_bundle


class StudioExecutionAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.workspace = root / "workspace"
        self.repository = self.workspace / "example"
        self.repository.mkdir(parents=True)
        (self.repository / "sample.py").write_text("value = 1\n", encoding="utf-8")
        self.storage = StudioStorage(root / "studio.sqlite3")
        self.boundary = WorkspaceRepositories(self.workspace)
        now = utc_now()
        self.storage.create_run(
            StudioRun(
                id="preview",
                repository_id="example",
                task="Update sample.",
                status="queued",
                phase="queued",
                created_at=now,
                updated_at=now,
            )
        )
        self.request = StartRunRequest(
            repository_id="example",
            task="Update sample.",
            self_correct=True,
            max_correction_rounds=2,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _verified_result(self) -> PlanExecutionResult:
        bundle = application_bundle()
        return PlanExecutionResult(
            plan=bundle.plan,
            candidate=bundle.candidate,
            status="verified",
            diff_text=bundle.diff_text,
            verification=PlanExecutionVerification(status="verified"),
            change_set_review=ChangeSetReview(
                overall_rating="good",
                summary="Candidate is ready.",
                file_results=[],
            ),
        )

    def test_preview_invokes_g1_g2_g3_and_stops_before_mutation(self) -> None:
        plan = application_bundle().plan

        def planning(*_args, event_sink=None, **_kwargs):
            event_sink("run_task_repository_exploration", {})
            event_sink("validate_engineering_plan", {})
            return plan

        def execution(*_args, event_sink=None, **kwargs):
            self.assertEqual(kwargs["max_correction_rounds"], 2)
            event_sink("generate_multi_file_candidate", {})
            event_sink("verify_candidate", {})
            event_sink("review_candidate_change_set", {})
            return self._verified_result()

        adapter = StudioExecutionAdapter(self.storage, self.boundary)
        with (
            patch("studio_backend.execution.plan_repository_task", side_effect=planning) as g1,
            patch("studio_backend.execution.execute_engineering_plan", side_effect=execution) as g23,
            patch(
                "studio_backend.execution.build_plan_application_bundle",
                return_value=application_bundle(),
            ) as build_bundle,
        ):
            adapter.execute("preview", self.request)
        self.assertEqual((g1.call_count, g23.call_count, build_bundle.call_count), (1, 1, 1))
        run = self.storage.get_run("preview")
        self.assertEqual((run.status, run.phase, run.approval_digest), ("verified", "approval_apply", DIGEST))
        self.assertEqual(
            (self.repository / "sample.py").read_text(encoding="utf-8"),
            "value = 1\n",
        )
        self.assertIsNone(self.storage.get_artifact("preview", "application_result"))
        self.assertIsNone(self.storage.get_artifact("preview", "local_git_delivery_result"))
        self.assertIsNone(self.storage.get_artifact("preview", "remote_delivery_result"))

    def test_failed_preview_cannot_build_application_bundle(self) -> None:
        result = self._verified_result().model_copy(update={"status": "failed"})
        adapter = StudioExecutionAdapter(self.storage, self.boundary)
        with (
            patch("studio_backend.execution.plan_repository_task", return_value=result.plan),
            patch("studio_backend.execution.execute_engineering_plan", return_value=result),
            patch("studio_backend.execution.build_plan_application_bundle") as build,
        ):
            adapter.execute("preview", self.request)
        build.assert_not_called()
        self.assertEqual(self.storage.get_run("preview").status, "failed")

    def test_self_correction_disabled_forces_zero_rounds(self) -> None:
        request = self.request.model_copy(
            update={"self_correct": False, "max_correction_rounds": 5}
        )
        observed: dict[str, int] = {}

        def execution(*_args, **kwargs):
            observed["rounds"] = kwargs["max_correction_rounds"]
            return self._verified_result()

        with (
            patch(
                "studio_backend.execution.plan_repository_task",
                return_value=application_bundle().plan,
            ),
            patch("studio_backend.execution.execute_engineering_plan", side_effect=execution),
            patch(
                "studio_backend.execution.build_plan_application_bundle",
                return_value=application_bundle(),
            ),
        ):
            StudioExecutionAdapter(self.storage, self.boundary).execute(
                "preview",
                request,
            )
        self.assertEqual(observed["rounds"], 0)

    def test_trace_and_candidate_lineage_are_persisted_outside_repository(self) -> None:
        bundle = application_bundle()
        corrected = bundle.candidate.model_copy(
            update={"summary": "Corrected deterministic candidate."}
        )

        def execution(*_args, attempt_artifact_sink=None, **_kwargs):
            attempt_artifact_sink(1, bundle.candidate)
            attempt_artifact_sink(2, corrected)
            return self._verified_result().model_copy(update={"candidate": corrected})

        with (
            patch(
                "studio_backend.execution.plan_repository_task",
                return_value=bundle.plan,
            ),
            patch(
                "studio_backend.execution.execute_engineering_plan",
                side_effect=execution,
            ),
            patch(
                "studio_backend.execution.build_plan_application_bundle",
                return_value=bundle,
            ),
        ):
            StudioExecutionAdapter(self.storage, self.boundary).execute(
                "preview", self.request
            )

        trace = self.storage.trace_sink.get_run("preview")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.status, "completed")
        artifacts = self.storage.trace_sink.list_artifacts("preview")
        edges = self.storage.trace_sink.list_edges("preview")
        self.assertEqual(sum(item.kind == "candidate" for item in artifacts), 2)
        self.assertIn("corrected_from", {edge.relation for edge in edges})
        self.assertIn("verified_by", {edge.relation for edge in edges})
        self.assertIn("reviewed_by", {edge.relation for edge in edges})
        self.assertIn("packaged_as", {edge.relation for edge in edges})
        self.assertFalse((self.repository / "trace-artifacts").exists())
