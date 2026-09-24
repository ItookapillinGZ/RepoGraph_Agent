"""H2.3 candidate provenance and exact regrade regressions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.candidate_artifacts import (
    CandidateArtifactError,
    CandidateArtifactStore,
    candidate_sha256,
    diff_sha256,
)
from evaluation.evaluator import build_local_evaluator_specification
from evaluation.models import EvaluationConfig
from evaluation.runner import run_evaluation_task
from evaluation.snapshot_regrade import regrade_result_candidate_versions
from tests.evaluation_harness.helpers import (
    FIXED,
    WRONG,
    StaticAdapter,
    candidate,
    make_repository,
    task,
)


class CandidateIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.commit = make_repository(self.root)
        self.workspace = self.root / "workspaces"
        self.artifacts = self.root / "artifacts"
        self.config = EvaluationConfig(name="h2-3", agentic_test=False)
        self.evaluation_task = task(self.source, self.commit)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_candidate_pair(self, first: str, final: str):
        return run_evaluation_task(
            self.evaluation_task,
            self.config,
            workspace_root=str(self.workspace),
            artifacts_root=str(self.artifacts),
            experiment_id="h2-3-test",
            adapter=StaticAdapter(
                candidate(final),
                first=candidate(first),
                rounds=1,
            ),
        )

    def test_candidate_v1_and_corrected_candidate_are_stored(self) -> None:
        result = self.run_candidate_pair(WRONG, FIXED)
        self.assertEqual(len(result.candidate_snapshot_paths), 2)
        store = CandidateArtifactStore(self.artifacts)
        first = store.load(result.candidate_snapshot_paths[0])
        final = store.load(result.candidate_snapshot_paths[1])
        self.assertEqual(first.attempt, 1)
        self.assertEqual(first.correction_round, 0)
        self.assertEqual(final.attempt, 2)
        self.assertEqual(final.correction_round, 1)
        self.assertEqual(first.candidate.files[0].content, WRONG)
        self.assertEqual(final.candidate.files[0].content, FIXED)
        self.assertEqual(result.first_candidate_sha256, first.candidate_sha256)
        self.assertEqual(result.final_candidate_sha256, final.candidate_sha256)

    def test_candidate_and_diff_hashes_are_stable(self) -> None:
        value = candidate(FIXED)
        self.assertEqual(candidate_sha256(value), candidate_sha256(value.model_copy()))
        self.assertEqual(diff_sha256("diff\n"), diff_sha256("diff\n"))

    def test_tampered_candidate_is_rejected_before_regrade(self) -> None:
        result = self.run_candidate_pair(WRONG, FIXED)
        path = self.artifacts / result.candidate_snapshot_paths[0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["candidate"]["summary"] = "tampered"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(CandidateArtifactError, "candidate_sha256"):
            CandidateArtifactStore(self.artifacts).load(
                result.candidate_snapshot_paths[0]
            )

    def test_first_and_final_regrade_are_exact_and_rescue_is_exact(self) -> None:
        result = self.run_candidate_pair(WRONG, FIXED)
        analysis = regrade_result_candidate_versions(
            self.evaluation_task,
            result,
            artifacts_root=self.artifacts,
            workspace_root=self.root / "regrade",
            evaluator_specification=result.evaluator_specification,
        )
        self.assertFalse(analysis.first_attempt_resolved)
        self.assertTrue(analysis.final_resolved)
        self.assertTrue(analysis.rescued_by_correction)
        self.assertFalse(analysis.regressed_after_correction)

    def test_regression_after_correction_is_exact(self) -> None:
        result = self.run_candidate_pair(FIXED, WRONG)
        analysis = regrade_result_candidate_versions(
            self.evaluation_task,
            result,
            artifacts_root=self.artifacts,
            workspace_root=self.root / "regrade",
            evaluator_specification=result.evaluator_specification,
        )
        self.assertTrue(analysis.first_attempt_resolved)
        self.assertFalse(analysis.final_resolved)
        self.assertFalse(analysis.rescued_by_correction)
        self.assertTrue(analysis.regressed_after_correction)

    def test_different_evaluator_regrade_is_marked(self) -> None:
        result = self.run_candidate_pair(WRONG, FIXED)
        different = build_local_evaluator_specification(
            self.evaluation_task,
            timeout_seconds=self.config.evaluator_timeout_seconds + 1,
        )
        analysis = regrade_result_candidate_versions(
            self.evaluation_task,
            result,
            artifacts_root=self.artifacts,
            workspace_root=self.root / "regrade",
            evaluator_specification=different,
        )
        self.assertTrue(analysis.first.regraded_with_different_evaluator)
        self.assertTrue(analysis.final.regraded_with_different_evaluator)


if __name__ == "__main__":
    unittest.main()
