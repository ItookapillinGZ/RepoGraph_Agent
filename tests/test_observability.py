"""H3.1 observability, lineage, privacy, ordering, and replay tests."""

from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import ValidationError

from evaluation.artifact_security import scan_persisted_artifacts
from observability.cli import _replay_frozen_h2
from observability.cli import main as observability_main
from observability.lineage import render_lineage
from observability.models import ReplayObservation, TraceRun
from observability.recorder import ObservabilityCallback, TraceRecorder, trace_run
from observability.redaction import REDACTED, TelemetryRedactionError, sanitize_metadata
from observability.replay import ReplayBlockedError, replay_artifact
from observability.sinks import CompositeTraceSink, NullTraceSink
from observability.storage import (
    SQLiteTraceSink,
    TraceIntegrityError,
)
from sandbox.models import SandboxExecutionResult, SandboxProvenance
from sandbox.policy import SandboxPolicy
from sandbox.runner import SandboxRunner


class ObservabilityFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.storage = SQLiteTraceSink(
            self.root / "trace.sqlite3", self.root / "artifacts"
        )
        self.recorder = TraceRecorder(self.storage, required=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def traced(self):
        return trace_run(
            self.recorder,
            run_kind="test_run",
            repo_identity="repo:fixture",
            task="private task text",
            task_summary="bounded summary",
            run_id="run-1",
        )


class ModelAndStorageTests(ObservabilityFixture):
    def test_models_are_strict(self) -> None:
        with self.assertRaises(ValidationError):
            TraceRun(
                run_id="x",
                run_kind="test",
                started_at="2026-01-01T00:00:00Z",
                status="active",
                root_span_id="root",
                unexpected=True,
            )

    def test_run_nested_spans_events_and_metrics_persist(self) -> None:
        with self.traced():
            with self.recorder.span("planning", kind="planning"):
                self.recorder.event(
                    "plan", kind="graph_node", metadata={"node": "plan"}
                )
            with self.recorder.span("verification", kind="verification"):
                self.recorder.event("verified", kind="verification")
        run = self.storage.get_run("run-1")
        self.assertIsNotNone(run)
        self.assertEqual(run.status, "completed")
        spans = self.storage.list_spans("run-1")
        self.assertEqual([item.name for item in spans], ["test_run", "planning", "verification"])
        self.assertEqual(spans[1].parent_span_id, spans[0].span_id)
        self.assertEqual(spans[2].parent_span_id, spans[0].span_id)
        sequences = [item.sequence_number for item in self.storage.list_events("run-1")]
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(len(sequences), len(set(sequences)))

    def test_sqlite_wal_foreign_keys_and_indexes(self) -> None:
        connection = sqlite3.connect(self.root / "trace.sqlite3")
        try:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                )
            }
            self.assertIn("idx_events_run_sequence", names)
            self.assertIn("idx_artifacts_run_kind", names)
        finally:
            connection.close()

    def test_concurrent_events_have_authoritative_unique_sequence(self) -> None:
        with self.traced():
            context = self.recorder.start_run  # keep the object alive for assertion clarity
            del context
            from observability.context import (
                current_trace_context,
                reset_trace_context,
                set_trace_context,
            )

            parent = current_trace_context()
            self.assertIsNotNone(parent)

            def append(index: int) -> None:
                token = set_trace_context(parent)
                try:
                    TraceRecorder(self.storage, required=True).event(
                        f"event-{index}", kind="test"
                    )
                finally:
                    reset_trace_context(token)

            workers = [threading.Thread(target=append, args=(index,)) for index in range(40)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
        sequences = [item.sequence_number for item in self.storage.list_events("run-1")]
        self.assertEqual(sequences, list(range(min(sequences), max(sequences) + 1)))
        self.assertEqual(len(sequences), 40)

    def test_crash_recovery_marks_only_active_records_interrupted(self) -> None:
        context = self.recorder.start_run(run_kind="crash", run_id="run-1")
        del context
        runs, spans = self.storage.reconcile_abandoned()
        self.assertEqual((runs, spans), (1, 1))
        self.assertEqual(self.storage.get_run("run-1").status, "interrupted")
        self.assertEqual(self.storage.list_spans("run-1")[0].status, "interrupted")

    def test_event_identity_is_idempotent_or_conflicting(self) -> None:
        with self.traced():
            event = self.recorder.event(
                "same", kind="test", event_id="event-fixed", metadata={"value": 1}
            )
            self.storage.append_event(event)
            with self.assertRaises(TraceIntegrityError):
                self.recorder.event(
                    "different",
                    kind="test",
                    event_id="event-fixed",
                    metadata={"value": 2},
                )

    def test_deterministic_export_is_stable(self) -> None:
        with self.traced():
            self.recorder.event("one", kind="test")
        first = json.dumps(self.storage.export_run("run-1"), sort_keys=True)
        second = json.dumps(self.storage.export_run("run-1"), sort_keys=True)
        self.assertEqual(first, second)


class RedactionTests(unittest.TestCase):
    def test_known_secret_forms_are_removed_before_persistence(self) -> None:
        fake_openai = "sk-abcdefghijklmnopqrstuvwxyz123456"
        fake_github = "ghp_abcdefghijklmnopqrstuvwxyz123456"
        environment = {
            "OPENAI_API_KEY": fake_openai,
            "GITHUB_TOKEN": fake_github,
        }
        payload = sanitize_metadata(
            {
                "api_key": fake_openai,
                "github": fake_github,
                "authorization": "Bearer secret-token-value",
                "proxy": "https://user:password@example.invalid",
                "dotenv": f"OPENAI_API_KEY={fake_openai}",
                "docker_auth_token": "docker-secret-value",
            },
            environment,
        )
        encoded = json.dumps(payload)
        self.assertNotIn(fake_openai, encoded)
        self.assertNotIn(fake_github, encoded)
        self.assertNotIn("secret-token-value", encoded)
        self.assertNotIn("password", encoded)
        self.assertIn(REDACTED, encoded)

    def test_absolute_user_path_is_portable(self) -> None:
        payload = sanitize_metadata({"path": r"C:\Users\alice\repo\src\a.py"}, {})
        self.assertNotIn("alice", str(payload))
        self.assertIn("<user-home>", str(payload))

    def test_oversized_metadata_is_rejected(self) -> None:
        with self.assertRaises(TelemetryRedactionError):
            sanitize_metadata({"value": "x" * 20_000}, {})

    def test_sensitive_artifact_is_rejected_instead_of_mutated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            storage = SQLiteTraceSink(root / "trace.sqlite3", root / "artifacts")
            required = TraceRecorder(storage, required=True)
            with (
                trace_run(required, run_kind="privacy", run_id="required"),
                self.assertRaises(TelemetryRedactionError),
            ):
                required.artifact(
                    kind="candidate", content="OPENAI_API_KEY=sk-fake-secret-value"
                )
            best_effort = TraceRecorder(storage, required=False)
            with trace_run(best_effort, run_kind="privacy", run_id="best-effort"):
                artifact = best_effort.artifact(
                    kind="candidate", content="OPENAI_API_KEY=sk-fake-secret-value"
                )
            self.assertIsNone(artifact)
            self.assertEqual(storage.list_artifacts("best-effort"), [])
            self.assertIn("TelemetryRedactionError:artifact_privacy", best_effort.failures)


class FailurePolicyTests(unittest.TestCase):
    class FailingSink(NullTraceSink):
        def append_event(self, event):
            raise OSError("disk unavailable")

    def test_best_effort_sink_records_failure_and_continues(self) -> None:
        sink = CompositeTraceSink([self.FailingSink()], required=False)
        recorder = TraceRecorder(sink, required=False)
        with trace_run(recorder, run_kind="best-effort", run_id="best-effort"):
            event = recorder.event("still-running", kind="test")
        self.assertIsNotNone(event)
        self.assertEqual(sink.failures, ["OSError:append_event"])

    def test_required_sink_propagates_failure(self) -> None:
        sink = CompositeTraceSink([self.FailingSink()], required=True)
        recorder = TraceRecorder(sink, required=True)
        with (
            self.assertRaisesRegex(OSError, "disk unavailable"),
            trace_run(recorder, run_kind="required", run_id="required"),
        ):
            recorder.event("must-fail", kind="test")


class PersistedSecurityScanTests(ObservabilityFixture):
    def test_trace_export_metadata_and_replay_report_are_secret_free(self) -> None:
        fake_openai = "sk-abcdefghijklmnopqrstuvwxyz123456"
        fake_github = "ghp_abcdefghijklmnopqrstuvwxyz123456"
        environment = {
            "OPENAI_API_KEY": fake_openai,
            "GITHUB_TOKEN": fake_github,
        }
        with self.traced():
            self.recorder.event(
                "redaction-check",
                kind="security",
                metadata={
                    "api_key": fake_openai,
                    "authorization": "Bearer fake-authorization-value",
                    "dotenv": f"GITHUB_TOKEN={fake_github}",
                },
            )
            artifact = self.recorder.artifact(
                kind="candidate",
                content={"patch": "safe"},
                metadata={
                    "api_key": fake_openai,
                    "github_token": fake_github,
                    "base_identity": "base-commit",
                    "verification_spec_digest": "a" * 64,
                    "evaluator_digest": "b" * 64,
                    "sandbox_backend": "docker",
                    "sandbox_image_id": "sha256:frozen",
                    "original_result": {
                        "status": "passed",
                        "resolved": True,
                        "exit_classification": "tests_passed",
                    },
                },
            )
        self.assertIsNotNone(artifact)
        replay = replay_artifact(
            self.storage,
            artifact.artifact_id,
            executor=lambda *_: ReplayObservation(
                status="passed", resolved=True, exit_classification="tests_passed"
            ),
            current_base_identity="base-commit",
            current_backend="docker",
            current_image_id="sha256:frozen",
        )
        export_path = self.root / "trace-export.jsonl"
        export_path.write_text(
            "\n".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                for record in self.storage.export_run("run-1")
            )
            + "\n",
            encoding="utf-8",
        )
        replay_path = self.root / "replay-report.json"
        replay_path.write_text(replay.model_dump_json(indent=2), encoding="utf-8")
        scan = scan_persisted_artifacts(
            [
                self.root / "trace.sqlite3",
                export_path,
                self.root / "artifacts",
                replay_path,
            ],
            environment=environment,
        )
        self.assertFalse(scan.secret_detected)
        self.assertGreaterEqual(scan.files_scanned, 3)


class ObservabilityCliTests(unittest.TestCase):
    def test_inspection_lineage_export_and_reconcile_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            storage = SQLiteTraceSink(
                root / "observability.sqlite3", root / "artifacts"
            )
            recorder = TraceRecorder(storage, required=True)
            with trace_run(recorder, run_kind="cli-test", run_id="cli-run"):
                parent = recorder.artifact(kind="candidate", content={"value": 1})
                child = recorder.artifact(kind="review_result", content={"value": 2})
                recorder.link(parent, child, "reviewed_by")
            export_path = root / "export.jsonl"
            commands = [
                ["runs"],
                ["show", "cli-run"],
                ["artifacts", "cli-run"],
                ["lineage", "cli-run"],
                ["export", "cli-run", "--output", str(export_path)],
                ["reconcile"],
            ]
            for command in commands:
                with self.subTest(command=command), redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        observability_main(["--state-root", str(root), *command]), 0
                    )
            first = export_path.read_text(encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    observability_main(
                        [
                            "--state-root",
                            str(root),
                            "export",
                            "cli-run",
                            "--output",
                            str(export_path),
                        ]
                    ),
                    0,
                )
            self.assertEqual(first, export_path.read_text(encoding="utf-8"))

    def test_replay_candidate_must_belong_to_requested_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            storage = SQLiteTraceSink(
                root / "observability.sqlite3", root / "artifacts"
            )
            recorder = TraceRecorder(storage, required=True)
            with trace_run(recorder, run_kind="cli-test", run_id="owner"):
                artifact = recorder.artifact(
                    kind="candidate",
                    content={"candidate": {}},
                    metadata={"replay_adapter": "h2_frozen_snapshot"},
                )
            with self.assertRaisesRegex(
                ReplayBlockedError, "does not belong to the requested run"
            ):
                _replay_frozen_h2(
                    storage, artifact.artifact_id, expected_run_id="different-run"
                )


class TelemetryBoundaryTests(ObservabilityFixture):
    def test_llm_usage_and_model_are_observable_without_prompt_content(self) -> None:
        callback = ObservabilityCallback()
        secret_prompt = "private prompt that must not be stored"
        response = LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=AIMessage(
                            content="structured output",
                            usage_metadata={
                                "input_tokens": 11,
                                "output_tokens": 7,
                                "total_tokens": 18,
                            },
                        )
                    )
                ]
            ],
            llm_output={"model_name": "deterministic-model"},
        )
        with self.traced():
            callback.on_llm_start(
                {"id": ["test-provider", "FakeModel"], "name": "configured-model"},
                [secret_prompt],
                run_id="llm-1",
                invocation_params={"model": "configured-model"},
            )
            callback.on_llm_end(response, run_id="llm-1")
        spans = self.storage.list_spans("run-1")
        llm = next(item for item in spans if item.kind == "llm_call")
        self.assertEqual(llm.parent_span_id, self.storage.get_run("run-1").root_span_id)
        event = next(
            item for item in self.storage.list_events("run-1") if item.kind == "llm_call"
        )
        self.assertEqual(event.span_id, llm.span_id)
        self.assertEqual(event.metadata["total_tokens"], 18)
        self.assertEqual(event.metadata["resolved_model"], "deterministic-model")
        exported = json.dumps(self.storage.export_run("run-1"))
        self.assertNotIn(secret_prompt, exported)
        self.assertNotIn("structured output", exported)

    def test_tool_arguments_and_large_sandbox_output_are_digest_only(self) -> None:
        callback = ObservabilityCallback()
        provenance = SandboxProvenance(
            backend="host",
            sandboxed=False,
            network="host",
            read_only_root=False,
            cap_drop_all=False,
            no_new_privileges=False,
            timeout_seconds=5,
        )
        huge = "x" * 500_000
        result = SandboxExecutionResult(
            status="passed",
            exit_code=0,
            stdout=huge,
            stderr="",
            duration_seconds=0.1,
            backend="host",
            provenance=provenance,
        )
        with self.traced():
            callback.on_tool_start(
                {"name": "read_repository_file"},
                "sensitive file contents",
                run_id="tool-1",
            )
            callback.on_tool_end("large tool result", run_id="tool-1")
            with patch("sandbox.runner.HostSandboxBackend.run", return_value=result):
                SandboxRunner(SandboxPolicy()).run_repository(
                    argv=["python", "-V"],
                    repository_root=self.root,
                    timeout_seconds=5,
                    purpose="tests",
                )
        exported = json.dumps(self.storage.export_run("run-1"))
        self.assertNotIn("sensitive file contents", exported)
        self.assertNotIn("large tool result", exported)
        self.assertNotIn(huge[:100], exported)
        sandbox_event = next(
            item
            for item in self.storage.list_events("run-1")
            if item.name == "sandbox_execution_completed"
        )
        self.assertEqual(sandbox_event.metadata["stdout_size"], len(huge))
        self.assertEqual(len(sandbox_event.metadata["stdout_sha256"]), 64)


class ArtifactLineageTests(ObservabilityFixture):
    def test_immutable_artifacts_dag_and_integrity(self) -> None:
        with self.traced():
            plan = self.recorder.artifact(kind="engineering_plan", content={"plan": 1})
            first = self.recorder.artifact(kind="candidate", content={"candidate": 1})
            diff = self.recorder.artifact(kind="candidate_diff", content="diff-v1")
            verification = self.recorder.artifact(
                kind="verification_report", content={"status": "failed"}
            )
            review = self.recorder.artifact(kind="review_result", content={"rating": "bad"})
            second = self.recorder.artifact(kind="candidate", content={"candidate": 2})
            bundle = self.recorder.artifact(kind="application_bundle", content={"bundle": 1})
            self.recorder.link(plan, first, "derived_from")
            self.recorder.link(first, diff, "derived_from")
            self.recorder.link(first, verification, "verified_by")
            self.recorder.link(first, review, "reviewed_by")
            self.recorder.link(first, second, "corrected_from")
            self.recorder.link(second, bundle, "packaged_as")
        rendered = render_lineage(
            self.storage.list_artifacts("run-1"), self.storage.list_edges("run-1")
        )
        self.assertIn("corrected_from", rendered)
        self.assertIn("verification_report", rendered)
        self.assertEqual(self.storage.metrics("run-1").candidate_count, 2)
        record, content = self.storage.load_artifact(first.artifact_id)
        path = self.storage.artifact_root / record.storage_ref
        path.write_bytes(content + b"tampered")
        with self.assertRaisesRegex(TraceIntegrityError, "artifact_integrity_error"):
            self.storage.load_artifact(first.artifact_id)

    def test_missing_parent_and_cycle_are_rejected(self) -> None:
        with self.traced():
            parent = self.recorder.artifact(kind="candidate", content="one")
            child = self.recorder.artifact(kind="candidate", content="two")
            self.recorder.link(parent, child, "corrected_from")
            with self.assertRaises(TraceIntegrityError):
                self.recorder.link(child, parent, "corrected_from")

    def test_same_content_can_have_distinct_semantic_identity(self) -> None:
        with self.traced():
            one = self.recorder.artifact(kind="candidate", content="same")
            two = self.recorder.artifact(kind="review_result", content="same")
        self.assertNotEqual(one.artifact_id, two.artifact_id)
        self.assertEqual(one.sha256, two.sha256)
        self.assertEqual(one.storage_ref, two.storage_ref)


class ReplayTests(ObservabilityFixture):
    def register_replay_candidate(
        self,
        *,
        backend: str = "docker",
        image: str | None = "sha256:frozen",
        resolved: bool = True,
    ):
        with self.traced():
            return self.recorder.artifact(
                kind="candidate",
                content={"patch": "immutable"},
                metadata={
                    "base_identity": "base-commit",
                    "verification_spec_digest": "a" * 64,
                    "evaluator_digest": "b" * 64,
                    "sandbox_backend": backend,
                    "sandbox_image_id": image,
                    "original_result": {
                        "status": "passed" if resolved else "failed",
                        "resolved": resolved,
                        "exit_classification": "tests_passed" if resolved else "assertion_failure",
                        "normalized_report_digest": None,
                    },
                },
            )

    def test_valid_replay_semantically_matches_without_agent_calls(self) -> None:
        artifact = self.register_replay_candidate()
        calls = {"verify": 0, "llm": 0, "mutation": 0}

        def verify(_content: bytes, _metadata: dict[str, object]) -> ReplayObservation:
            calls["verify"] += 1
            return ReplayObservation(
                status="passed", resolved=True, exit_classification="tests_passed"
            )

        result = replay_artifact(
            self.storage,
            artifact.artifact_id,
            executor=verify,
            current_base_identity="base-commit",
            current_backend="docker",
            current_image_id="sha256:frozen",
        )
        self.assertTrue(result.semantic_match)
        self.assertEqual(calls, {"verify": 1, "llm": 0, "mutation": 0})

    def test_semantic_mismatch_is_explicit(self) -> None:
        artifact = self.register_replay_candidate()
        result = replay_artifact(
            self.storage,
            artifact.artifact_id,
            executor=lambda *_: ReplayObservation(
                status="failed", resolved=False, exit_classification="assertion_failure"
            ),
            current_base_identity="base-commit",
            current_backend="docker",
            current_image_id="sha256:frozen",
        )
        self.assertFalse(result.semantic_match)
        self.assertTrue(result.warnings)

    def test_base_backend_image_and_docker_availability_fail_closed(self) -> None:
        artifact = self.register_replay_candidate()
        executor = lambda *_: ReplayObservation(status="passed", resolved=True)
        cases = [
            {"current_base_identity": "wrong", "current_backend": "docker", "current_image_id": "sha256:frozen"},
            {"current_base_identity": "base-commit", "current_backend": "host", "current_image_id": None},
            {"current_base_identity": "base-commit", "current_backend": "docker", "current_image_id": None},
            {"current_base_identity": "base-commit", "current_backend": "docker", "current_image_id": "sha256:drift"},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ReplayBlockedError):
                replay_artifact(
                    self.storage, artifact.artifact_id, executor=executor, **case
                )

    def test_hash_mismatch_stops_before_executor(self) -> None:
        artifact = self.register_replay_candidate()
        path = self.storage.artifact_root / artifact.storage_ref
        path.write_bytes(b"tampered")
        called = False

        def verify(*_args):
            nonlocal called
            called = True
            return ReplayObservation(status="passed", resolved=True)

        with self.assertRaises(TraceIntegrityError):
            replay_artifact(
                self.storage,
                artifact.artifact_id,
                executor=verify,
                current_base_identity="base-commit",
                current_backend="docker",
                current_image_id="sha256:frozen",
            )
        self.assertFalse(called)


class DeterministicGraphTraceTests(unittest.TestCase):
    def test_real_deterministic_preview_builds_nested_trace_without_api(self) -> None:
        from evaluation.models import EvaluationConfig
        from evaluation.runner import RealRepoGraphAdapter, run_evaluation_task
        from tests.evaluation_harness.helpers import (
            FIXED,
            candidate,
            make_repository,
            task,
        )
        from tests.evaluation_harness.test_integration import (
            execution_graph,
            planning_graph,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, commit = make_repository(root)
            storage = SQLiteTraceSink(root / "trace.sqlite3", root / "artifacts")
            adapter = RealRepoGraphAdapter(
                planning_graph=planning_graph(),
                execution_graph=execution_graph(candidate(FIXED)),
                trace_sink=storage,
                trace_required=True,
            )
            result = run_evaluation_task(
                task(source, commit),
                EvaluationConfig(
                    name="trace-integration",
                    agentic_explore=False,
                    agentic_test=False,
                ),
                workspace_root=str(root / "workspaces"),
                adapter=adapter,
            )
            self.assertEqual(result.status, "resolved")
            runs = storage.list_runs()
            self.assertEqual(len(runs), 1)
            spans = storage.list_spans(runs[0].run_id)
            by_name = {span.name: span for span in spans}
            self.assertIn("engineering_plan_graph", by_name)
            self.assertIn("repository_exploration_graph", by_name)
            self.assertIn("plan_execution_graph", by_name)
            self.assertIn("sandbox_execution", by_name)
            self.assertEqual(
                by_name["repository_exploration_graph"].parent_span_id,
                by_name["engineering_plan_graph"].span_id,
            )
            self.assertEqual(
                by_name["sandbox_execution"].parent_span_id,
                by_name["plan_execution_graph"].span_id,
            )
            self.assertTrue(all(span.status == "completed" for span in spans))


if __name__ == "__main__":
    unittest.main()
