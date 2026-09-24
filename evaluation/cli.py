"""Independent command line entry point for Stage H2 evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404
import sys
from pathlib import Path

from dotenv import load_dotenv

from evaluation.adapters.local_tasks import LocalTaskDataset
from evaluation.adapters.swebench import SWEbenchDataset, export_predictions
from evaluation.adapters.swebench_evaluator import (
    detect_swebench_prerequisites,
    evaluate_swebench_prediction,
)
from evaluation.campaign import (
    CampaignError,
    freeze_local_tasks,
    run_live_campaign,
)
from evaluation.campaign_h2_3 import (
    H2_3_PLANNED_BASE_TASK_RUNS,
    freeze_h2_3_tasks,
    run_h2_3_campaign,
)
from evaluation.campaign_h2_3c import MAX_H2_3C_LIVE_TASK_RUNS, run_h2_3c_campaign
from evaluation.campaign_h2_3d import MAX_H2_3D_LIVE_TASK_RUNS, run_h2_3d_campaign
from evaluation.campaign_h2_3e import MAX_H2_3E_LIVE_TASK_RUNS, run_h2_3e_campaign
from evaluation.campaign_integrity import (
    audit_campaign,
    load_history,
    reconcile_legacy_attempts,
)
from evaluation.campaign_regrade import run_campaign_regrade
from evaluation.campaign_resume import resume_live_campaign
from evaluation.fixtures import create_local_benchmark
from evaluation.h2_3_fixtures import create_h2_3_benchmark
from evaluation.live import (
    DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    MAX_H2_2_LIVE_TASK_RUNS,
    MAX_H2_3_LIVE_TASK_RUNS,
    LiveGuardError,
    LiveRunBudget,
    prepare_live_environment,
    require_live_flag,
    run_live_preflight,
)
from evaluation.models import (
    MAX_EVALUATOR_TIMEOUT_SECONDS,
    MAX_TASK_TIMEOUT_SECONDS,
    EvaluationConfig,
)
from evaluation.reports import render_summary, write_reports
from evaluation.runner import create_experiment, run_experiment
from evaluation.storage import EvaluationStorage
from model_defaults import (
    ModelConfigurationError,
    ProductionModelSettings,
    get_production_model_settings,
)


def _git_commit() -> str | None:
    completed = subprocess.run(  # nosec B603 B607
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
        shell=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _storage(root: Path) -> EvaluationStorage:
    return EvaluationStorage(root / "evaluation.db", root / "results")


def _bounded_timeout(maximum: float):
    def parse(value: str) -> float:
        try:
            timeout = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("timeout must be a number") from error
        if timeout <= 0 or timeout > maximum:
            raise argparse.ArgumentTypeError(
                f"timeout must be greater than 0 and at most {maximum:g} seconds"
            )
        return timeout

    return parse


def _bounded_positive_int(maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("value must be an integer") from error
        if parsed < 1 or parsed > maximum:
            raise argparse.ArgumentTypeError(f"value must be between 1 and {maximum}")
        return parsed

    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a sequential evaluation experiment")
    run.add_argument("--dataset", required=True)
    run.add_argument("--dataset-type", choices=("local", "swebench"), default="local")
    run.add_argument("--workspace", required=True)
    run.add_argument("--name", required=True)
    run.add_argument(
        "--self-correct", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument("--max-correction-rounds", type=int, default=2)
    run.add_argument(
        "--agentic-explore", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument(
        "--agentic-test", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument("--run-tests", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--sandbox", choices=("host", "docker"), default="host")
    run.add_argument("--model-name")
    run.add_argument("--max-tasks", type=int)
    run.add_argument(
        "--task-timeout",
        "--overall-timeout",
        dest="task_timeout",
        type=_bounded_timeout(MAX_TASK_TIMEOUT_SECONDS),
        default=DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    )
    run.add_argument(
        "--evaluator-timeout",
        type=_bounded_timeout(MAX_EVALUATOR_TIMEOUT_SECONDS),
        default=300,
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument("--rerun-failed", action="store_true")
    run.add_argument("--live", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--max-live-task-runs",
        type=_bounded_positive_int(MAX_H2_2_LIVE_TASK_RUNS),
        default=MAX_H2_2_LIVE_TASK_RUNS,
    )

    campaign = subparsers.add_parser(
        "campaign", help="Run the sequential H2.2 live local campaign"
    )
    campaign.add_argument("--dataset", required=True)
    campaign.add_argument("--workspace", required=True)
    campaign.add_argument("--output", required=True)
    campaign.add_argument("--live", action="store_true")
    campaign.add_argument("--dry-run", action="store_true")
    campaign.add_argument("--resume", action="store_true")
    campaign.add_argument(
        "--max-live-task-runs",
        type=_bounded_positive_int(MAX_H2_2_LIVE_TASK_RUNS),
        default=MAX_H2_2_LIVE_TASK_RUNS,
    )
    campaign.add_argument(
        "--task-timeout",
        type=_bounded_timeout(MAX_TASK_TIMEOUT_SECONDS),
        default=DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    )
    campaign.add_argument(
        "--evaluator-timeout",
        type=_bounded_timeout(MAX_EVALUATOR_TIMEOUT_SECONDS),
        default=300,
    )

    h2_3 = subparsers.add_parser(
        "campaign-h2-3",
        help="Run the sequential H2.3 controlled harder benchmark",
    )
    h2_3.add_argument("--dataset", required=True)
    h2_3.add_argument("--workspace", required=True)
    h2_3.add_argument("--output", required=True)
    h2_3.add_argument("--live", action="store_true")
    h2_3.add_argument("--dry-run", action="store_true")
    h2_3.add_argument(
        "--max-live-task-runs",
        type=_bounded_positive_int(MAX_H2_3_LIVE_TASK_RUNS),
        default=MAX_H2_3_LIVE_TASK_RUNS,
    )
    h2_3.add_argument(
        "--task-timeout",
        type=_bounded_timeout(MAX_TASK_TIMEOUT_SECONDS),
        default=DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    )
    h2_3.add_argument(
        "--evaluator-timeout",
        type=_bounded_timeout(MAX_EVALUATOR_TIMEOUT_SECONDS),
        default=300,
    )
    h2_3c = subparsers.add_parser(
        "campaign-h2-3c",
        help="Append-only completion of the frozen H2.3 ablations",
    )
    h2_3c.add_argument("--dataset", required=True)
    h2_3c.add_argument("--parent-workspace", required=True)
    h2_3c.add_argument("--workspace", required=True)
    h2_3c.add_argument("--parent-output", required=True)
    h2_3c.add_argument("--output", required=True)
    h2_3c.add_argument("--source-root", default=str(Path.cwd()))
    h2_3c.add_argument("--live", action="store_true")
    h2_3c.add_argument("--dry-run", action="store_true")
    h2_3d = subparsers.add_parser(
        "campaign-h2-3d",
        help="Run the clean direct-OpenAI gpt-5.6-luna paired campaign",
    )
    h2_3d.add_argument("--dataset", required=True)
    h2_3d.add_argument("--workspace", required=True)
    h2_3d.add_argument("--output", required=True)
    h2_3d.add_argument("--frozen-output", required=True)
    h2_3d.add_argument("--source-root", default=str(Path.cwd()))
    h2_3d.add_argument("--live", action="store_true")
    h2_3d.add_argument("--dry-run", action="store_true")
    h2_3d.add_argument(
        "--task-timeout",
        type=_bounded_timeout(MAX_TASK_TIMEOUT_SECONDS),
        default=DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    )
    h2_3d.add_argument(
        "--evaluator-timeout",
        type=_bounded_timeout(MAX_EVALUATOR_TIMEOUT_SECONDS),
        default=300,
    )
    h2_3e = subparsers.add_parser(
        "campaign-h2-3e",
        help="Recover H2.3D attempt identities and complete missing Luna ablations",
    )
    h2_3e.add_argument("--dataset", required=True)
    h2_3e.add_argument("--workspace", required=True)
    h2_3e.add_argument("--output", required=True)
    h2_3e.add_argument("--parent-output", required=True)
    h2_3e.add_argument("--frozen-output", required=True)
    h2_3e.add_argument("--source-root", default=str(Path.cwd()))
    h2_3e.add_argument("--live", action="store_true")
    h2_3e.add_argument("--dry-run", action="store_true")
    h2_3e.add_argument(
        "--task-timeout",
        type=_bounded_timeout(MAX_TASK_TIMEOUT_SECONDS),
        default=DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    )
    h2_3e.add_argument(
        "--evaluator-timeout",
        type=_bounded_timeout(MAX_EVALUATOR_TIMEOUT_SECONDS),
        default=300,
    )
    regrade = subparsers.add_parser(
        "campaign-regrade",
        help="Offline-regrade immutable H2.2 prediction patches",
    )
    regrade.add_argument("--workspace", required=True)
    regrade.add_argument("--campaign-output", required=True)
    regrade.add_argument("--regrade-workspace", required=True)

    audit = subparsers.add_parser(
        "audit-campaign",
        help="Read-only audit of campaign attempts, provenance, and snapshots",
    )
    audit.add_argument("--workspace", required=True)
    audit.add_argument("--artifacts", required=True)
    audit.add_argument("--campaign-id", required=True)
    audit.add_argument("--experiment-id", action="append")
    audit.add_argument("--lineage-campaign-id", action="append")

    reconcile = subparsers.add_parser(
        "reconcile-campaign",
        help="Explicitly import legacy reservation/history attempt identities",
    )
    reconcile.add_argument("--workspace", required=True)
    reconcile.add_argument("--campaign-id", required=True)
    reconcile.add_argument("--budget-ledger", required=True)
    reconcile.add_argument("--history", required=True)

    reconcile_incomplete = subparsers.add_parser(
        "reconcile-incomplete",
        help="Mark stale reserved/running attempts interrupted before a restart",
    )
    reconcile_incomplete.add_argument("--workspace", required=True)
    reconcile_incomplete.add_argument("--campaign-id", required=True)
    reconcile_incomplete.add_argument(
        "--reason", default="Explicit controller restart reconciliation."
    )

    report = subparsers.add_parser("report", help="Generate experiment reports")
    report.add_argument("--experiment", required=True)
    report.add_argument("--workspace", required=True)

    export = subparsers.add_parser(
        "swebench-export",
        aliases=["export-swebench"],
        help="Export official prediction JSONL fields",
    )
    export.add_argument("--experiment", required=True)
    export.add_argument("--workspace", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--model-name-or-path", required=True)

    official = subparsers.add_parser(
        "swebench-evaluate",
        help="Grade one prediction with the official SWE-bench Docker harness",
    )
    official.add_argument("--predictions", required=True)
    official.add_argument("--dataset", required=True)
    official.add_argument("--split", default="test")
    official.add_argument("--instance-id", required=True)
    official.add_argument("--run-id", required=True)
    official.add_argument("--output", required=True)
    official.add_argument(
        "--timeout",
        type=_bounded_timeout(MAX_EVALUATOR_TIMEOUT_SECONDS),
        default=1_800,
    )

    subparsers.add_parser(
        "doctor",
        help="Check optional SWE-bench and Docker prerequisites",
    )

    fixtures = subparsers.add_parser(
        "create-local-fixtures", help="Generate the ten-task local-v1 benchmark"
    )
    fixtures.add_argument("--output", required=True)
    harder_fixtures = subparsers.add_parser(
        "create-h2-3-fixtures",
        help="Generate the 20-task H2.3 benchmark with external hidden tests",
    )
    harder_fixtures.add_argument("--output", required=True)
    return parser


def _configured_model_settings(*, live: bool) -> ProductionModelSettings:
    if live:
        return prepare_live_environment()
    load_dotenv()
    try:
        return get_production_model_settings()
    except ModelConfigurationError as error:
        raise LiveGuardError(str(error)) from error


def _print_run_summary(
    *,
    live: bool,
    settings: ProductionModelSettings,
    task_count: int,
    configs: int,
    maximum_task_runs: int,
    timeout_seconds: float,
) -> None:
    print("Live LLM calls: " + ("ENABLED" if live else "DISABLED (dry run)"))
    print(f"Provider: {settings.provider}")
    print(f"Model: {settings.model}")
    api_base = settings.base_url or "OpenAI SDK default"
    print(f"API base URL: {api_base}")
    print(f"Tasks: {task_count}")
    print(f"Configs: {configs}")
    print(f"Maximum task-runs: {maximum_task_runs}")
    print(f"Task timeout: {timeout_seconds:g}s")


def _run(args: argparse.Namespace) -> int:
    require_live_flag(live=args.live, dry_run=args.dry_run)
    settings = _configured_model_settings(live=args.live)
    if args.model_name and args.model_name != settings.model:
        raise SystemExit(
            "The production API does not expose a model override; use the configured model."
        )
    root = Path(args.workspace).resolve()
    adapter = (
        LocalTaskDataset(args.dataset)
        if args.dataset_type == "local"
        else SWEbenchDataset(args.dataset)
    )
    tasks = list(adapter.load())
    selected_count = min(len(tasks), args.max_tasks or len(tasks))
    _print_run_summary(
        live=args.live,
        settings=settings,
        task_count=selected_count,
        configs=1,
        maximum_task_runs=args.max_live_task_runs,
        timeout_seconds=args.task_timeout,
    )
    if args.dry_run:
        return 0
    print("LLM API key: configured")
    preflight = run_live_preflight(settings=settings)
    print(f"Preflight LLM calls: {preflight.llm_calls}")
    root.mkdir(parents=True, exist_ok=True)
    storage = _storage(root)
    budget = LiveRunBudget(
        root / "live-task-runs.jsonl",
        maximum=args.max_live_task_runs,
    )
    if budget.consumed + selected_count > args.max_live_task_runs:
        raise SystemExit("Requested run exceeds the remaining live task-run budget.")
    config = EvaluationConfig(
        name=args.name,
        self_correct=args.self_correct,
        max_correction_rounds=args.max_correction_rounds,
        agentic_explore=args.agentic_explore,
        agentic_test=args.agentic_test,
        run_tests=args.run_tests,
        sandbox_backend=args.sandbox,
        model_name=settings.model,
        model_temperature=settings.temperature,
        model_seed=settings.seed,
        model_provider=settings.provider,
        model_base_url=settings.base_url,
        model_reasoning_effort=settings.reasoning_effort,
        model_max_completion_tokens=settings.max_completion_tokens,
        llm_execution_kind="live",
        max_tasks=args.max_tasks,
        task_timeout_seconds=args.task_timeout,
        evaluator_timeout_seconds=args.evaluator_timeout,
    )
    existing = storage.get_experiment(args.name)
    if args.resume or args.rerun_failed:
        if existing is None:
            raise SystemExit(f"Experiment not found for resume: {args.name}")
        if existing.llm_execution_kind != "live":
            raise SystemExit("Cannot resume a non-live experiment as live.")
        experiment = existing
    else:
        experiment = create_experiment(
            args.name,
            f"{args.dataset_type}:{Path(args.dataset).resolve()}",
            config,
            git_commit=_git_commit(),
        )
    try:
        results = run_experiment(
            tasks,
            experiment,
            workspace_root=str(root / "workspaces"),
            storage=storage,
            rerun_failed=args.rerun_failed,
            before_task_run=lambda current, task: budget.reserve(
                experiment_id=current.id,
                task_id=task.id,
                config_name=current.config.name,
            ),
        )
    except KeyboardInterrupt:
        print(
            "Evaluation interrupted; completed tasks remain persisted.", file=sys.stderr
        )
        return 130
    write_reports(experiment, results, root / "reports")
    print(render_summary(experiment, results))
    return 0


def _campaign(args: argparse.Namespace) -> int:
    require_live_flag(live=args.live, dry_run=args.dry_run)
    if args.resume and not args.live:
        raise LiveGuardError("Campaign resume requires explicit --live")

    settings = _configured_model_settings(live=args.live)
    tasks = freeze_local_tasks(LocalTaskDataset(args.dataset).load())
    _print_run_summary(
        live=args.live,
        settings=settings,
        task_count=len(tasks),
        configs=4,
        maximum_task_runs=44,
        timeout_seconds=args.task_timeout,
    )
    if args.dry_run:
        return 0
    print("LLM API key: configured")
    campaign_runner = resume_live_campaign if args.resume else run_live_campaign
    outcome = campaign_runner(
        tasks,
        dataset=f"local:{Path(args.dataset).resolve()}",
        workspace_root=args.workspace,
        output_root=args.output,
        maximum_live_task_runs=args.max_live_task_runs,
        task_timeout_seconds=args.task_timeout,
        evaluator_timeout_seconds=args.evaluator_timeout,
        repograph_commit=_git_commit(),
        model_settings=settings,
    )
    print(f"Manifest: {outcome.manifest_path}")
    print(f"Live task-runs consumed: {outcome.live_task_runs_consumed}")
    return 0


def _campaign_h2_3(args: argparse.Namespace) -> int:
    require_live_flag(live=args.live, dry_run=args.dry_run)
    settings = _configured_model_settings(live=args.live)
    all_tasks = LocalTaskDataset(args.dataset).load()
    formal, smoke = freeze_h2_3_tasks(all_tasks)
    _print_run_summary(
        live=args.live,
        settings=settings,
        task_count=len(formal),
        configs=4,
        maximum_task_runs=args.max_live_task_runs,
        timeout_seconds=args.task_timeout,
    )
    print(f"Smoke task-runs: {len(smoke)}")
    print(f"Planned base task-runs: {H2_3_PLANNED_BASE_TASK_RUNS}")
    if args.dry_run:
        return 0
    print("LLM API key: configured")
    outcome = run_h2_3_campaign(
        all_tasks,
        dataset=f"local:{Path(args.dataset).resolve()}",
        workspace_root=args.workspace,
        output_root=args.output,
        maximum_live_task_runs=args.max_live_task_runs,
        task_timeout_seconds=args.task_timeout,
        evaluator_timeout_seconds=args.evaluator_timeout,
        repograph_commit=_git_commit(),
        model_settings=settings,
    )
    print(f"Manifest: {outcome.manifest_path}")
    print(f"Live task-runs consumed: {outcome.live_task_runs_consumed}")
    return 0


def _campaign_h2_3c(args: argparse.Namespace) -> int:
    require_live_flag(live=args.live, dry_run=args.dry_run)
    settings = _configured_model_settings(live=args.live)
    print("Stage H2.3C: append-only continuation")
    print(f"Frozen model: {settings.model}")
    print(f"Maximum continuation task-runs: {MAX_H2_3C_LIVE_TASK_RUNS}")
    if args.dry_run:
        return 0
    print("LLM API key: configured")
    outcome = run_h2_3c_campaign(
        dataset_path=args.dataset,
        parent_workspace_root=args.parent_workspace,
        workspace_root=args.workspace,
        parent_output_root=args.parent_output,
        output_root=args.output,
        source_root=args.source_root,
        model_settings=settings,
    )
    print(f"Manifest: {outcome['manifest']}")
    print(f"Final output: {outcome['final_output']}")
    print(f"H2.3C live task-runs consumed: {outcome['live_task_runs_consumed']}")
    return 0


def _campaign_h2_3d(args: argparse.Namespace) -> int:
    require_live_flag(live=args.live, dry_run=args.dry_run)
    settings = _configured_model_settings(live=args.live)
    all_tasks = LocalTaskDataset(args.dataset).load()
    formal, _smoke = freeze_h2_3_tasks(all_tasks)
    print("Live LLM calls: " + ("ENABLED" if args.live else "DISABLED (dry run)"))
    print(f"Provider adapter mode: {settings.provider}")
    print(f"Model: {settings.model}")
    print("API route: Direct OpenAI (sanitized)")
    print(f"Tasks: {len(formal)}")
    print("Configs: 4")
    print(f"Maximum task-runs: {MAX_H2_3D_LIVE_TASK_RUNS}")
    print(f"Task timeout: {args.task_timeout:g}s")
    print("Stage H2.3D: Direct OpenAI clean paired campaign")
    print("Planned base task-runs: 68")
    print("Infrastructure retry reserve: 12")
    if args.dry_run:
        return 0
    print("LLM API key: configured")
    outcome = run_h2_3d_campaign(
        dataset_path=args.dataset,
        workspace_root=args.workspace,
        output_root=args.output,
        frozen_output_root=args.frozen_output,
        source_root=args.source_root,
        model_settings=settings,
        task_timeout_seconds=args.task_timeout,
        evaluator_timeout_seconds=args.evaluator_timeout,
        repograph_commit=_git_commit(),
    )
    print(f"Manifest: {outcome['manifest']}")
    print(f"Output: {outcome['output']}")
    print(f"H2.3D live task-runs consumed: {outcome['live_task_runs_consumed']}")
    return 0


def _campaign_h2_3e(args: argparse.Namespace) -> int:
    require_live_flag(live=args.live, dry_run=args.dry_run)
    settings = _configured_model_settings(live=args.live)
    print("Stage H2.3E: interrupted-run recovery continuation")
    print(f"Provider adapter mode: {settings.provider}")
    print(f"Model: {settings.model}")
    print("API route: Direct OpenAI (sanitized)")
    print("Workers: 1")
    print(f"Maximum continuation task-runs: {MAX_H2_3E_LIVE_TASK_RUNS}")
    if args.dry_run:
        return 0
    print("LLM API key: configured")
    outcome = run_h2_3e_campaign(
        dataset_path=args.dataset,
        workspace_root=args.workspace,
        output_root=args.output,
        parent_output_root=args.parent_output,
        frozen_output_root=args.frozen_output,
        source_root=args.source_root,
        model_settings=settings,
        task_timeout_seconds=args.task_timeout,
        evaluator_timeout_seconds=args.evaluator_timeout,
        repograph_commit=_git_commit(),
    )
    print(f"Manifest: {outcome['manifest']}")
    print(f"Authoritative view: {outcome['authoritative_view']}")
    print(f"H2.3E live task-runs consumed: {outcome['live_task_runs_consumed']}")
    return 0


def _campaign_regrade(args: argparse.Namespace) -> int:
    print("LLM API calls: DISABLED (offline patch regrade)")
    outcome = run_campaign_regrade(
        storage_root=args.workspace,
        campaign_output_root=args.campaign_output,
        regrade_workspace_root=args.regrade_workspace,
    )
    print(f"Regraded predictions: {outcome.regraded_predictions}")
    print(f"Unavailable predictions: {outcome.unavailable_predictions}")
    print(f"Regrade manifest: {outcome.manifest_path}")
    print(f"Comparison: {outcome.comparison_path}")
    return 0


def _audit_campaign(args: argparse.Namespace) -> int:
    root = Path(args.workspace).resolve()
    report = audit_campaign(
        _storage(root),
        artifacts_root=args.artifacts,
        campaign_id=args.campaign_id,
        experiment_ids=args.experiment_id,
        lineage_campaign_ids=args.lineage_campaign_id,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if report.passed else 2


def _reconcile_campaign(args: argparse.Namespace) -> int:
    root = Path(args.workspace).resolve()
    budget = LiveRunBudget(args.budget_ledger, maximum=MAX_H2_3_LIVE_TASK_RUNS)
    records = reconcile_legacy_attempts(
        _storage(root),
        campaign_id=args.campaign_id,
        reservations=budget.reservations(),
        history_records=load_history(args.history),
    )
    print(f"Reconciled execution attempts: {len(records)}")
    return 0


def _reconcile_incomplete(args: argparse.Namespace) -> int:
    changed = _storage(Path(args.workspace).resolve()).reconcile_incomplete_attempts(
        campaign_id=args.campaign_id,
        reason=args.reason,
    )
    print(f"Interrupted stale execution attempts: {changed}")
    return 0


def _report(args: argparse.Namespace) -> int:
    root = Path(args.workspace).resolve()
    storage = _storage(root)
    experiment = storage.get_experiment(args.experiment)
    if experiment is None:
        raise SystemExit(f"Experiment not found: {args.experiment}")
    results = storage.list_results(experiment.id)
    markdown, structured = write_reports(experiment, results, root / "reports")
    print(render_summary(experiment, results))
    print(f"Markdown: {markdown}")
    print(f"JSON: {structured}")
    return 0


def _export(args: argparse.Namespace) -> int:
    storage = _storage(Path(args.workspace).resolve())
    experiment = storage.get_experiment(args.experiment)
    if experiment is None:
        raise SystemExit(f"Experiment not found: {args.experiment}")
    output = export_predictions(
        storage.list_results(experiment.id),
        args.output,
        model_name_or_path=args.model_name_or_path,
    )
    print(f"Predictions: {output}")
    return 0


def _swebench_evaluate(args: argparse.Namespace) -> int:
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    result = evaluate_swebench_prediction(
        args.predictions,
        dataset_name=args.dataset,
        split=args.split,
        instance_id=args.instance_id,
        experiment_id=args.run_id,
        timeout_seconds=args.timeout,
        output_root=root,
    )
    result_path = root / f"{result.run_id}.repograph-result.json"
    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Official SWE-bench status: {result.status}")
    print(f"Run ID: {result.run_id}")
    print(f"Result: {result_path}")
    return 0 if result.status in {"resolved", "unresolved"} else 2


def _doctor() -> int:
    prerequisites = detect_swebench_prerequisites()
    print(
        "SWE-bench installed: " + ("yes" if prerequisites.swebench_installed else "no")
    )
    print(
        "Docker CLI available: "
        + ("yes" if prerequisites.docker_cli_available else "no")
    )
    print(
        "Docker daemon available: "
        + ("yes" if prerequisites.docker_daemon_available else "no")
    )
    return 0 if prerequisites.ready else 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "campaign":
        return _campaign(args)
    if args.command == "campaign-h2-3":
        return _campaign_h2_3(args)
    if args.command == "campaign-h2-3c":
        return _campaign_h2_3c(args)
    if args.command == "campaign-h2-3d":
        return _campaign_h2_3d(args)
    if args.command == "campaign-h2-3e":
        return _campaign_h2_3e(args)
    if args.command == "campaign-regrade":
        return _campaign_regrade(args)
    if args.command == "audit-campaign":
        return _audit_campaign(args)
    if args.command == "reconcile-campaign":
        return _reconcile_campaign(args)
    if args.command == "reconcile-incomplete":
        return _reconcile_incomplete(args)
    if args.command == "report":
        return _report(args)
    if args.command in {"swebench-export", "export-swebench"}:
        return _export(args)
    if args.command == "swebench-evaluate":
        return _swebench_evaluate(args)
    if args.command == "doctor":
        return _doctor()
    if args.command == "create-local-fixtures":
        print(f"Dataset: {create_local_benchmark(args.output)}")
        return 0
    if args.command == "create-h2-3-fixtures":
        print(f"Dataset: {create_h2_3_benchmark(args.output)}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        exit_code = main()
    except (CampaignError, LiveGuardError) as error:
        print(f"Evaluation stopped: {error}", file=sys.stderr)
        exit_code = 2
    except KeyboardInterrupt:
        print(
            "Evaluation interrupted; persisted results were preserved.", file=sys.stderr
        )
        exit_code = 130
    raise SystemExit(exit_code)
