"""Stage H2.3E interrupted-run recovery and paired campaign completion."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from evaluation.adapters.local_tasks import LocalTaskDataset
from evaluation.artifact_security import scan_persisted_artifacts
from evaluation.campaign import CampaignError
from evaluation.campaign_h2_3d import (
    H2_3D_CAMPAIGN_ID,
    H23DCampaign,
)
from evaluation.campaign_integrity import audit_campaign
from evaluation.live import DEFAULT_LIVE_TASK_TIMEOUT_SECONDS, prepare_live_environment
from model_defaults import ProductionModelSettings

H2_3E_CAMPAIGN_ID = "h2_3e_openai_luna_recovery"
MAX_H2_3E_LIVE_TASK_RUNS = 45
H2_3E_PLANNED_BASE_TASK_RUNS = 31
H2_3E_INFRASTRUCTURE_RETRY_RESERVE = (
    MAX_H2_3E_LIVE_TASK_RUNS - H2_3E_PLANNED_BASE_TASK_RUNS
)


def _copy_authoritative_view(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)


def run_h2_3e_campaign(
    *,
    dataset_path: str | Path,
    workspace_root: str | Path,
    output_root: str | Path,
    parent_output_root: str | Path,
    frozen_output_root: str | Path,
    source_root: str | Path,
    model_settings: ProductionModelSettings | None = None,
    task_timeout_seconds: float = DEFAULT_LIVE_TASK_TIMEOUT_SECONDS,
    evaluator_timeout_seconds: float = 300.0,
    repograph_commit: str | None = None,
) -> dict[str, Any]:
    """Reuse audited completed configs and run only unfinished assignments."""

    settings = model_settings or prepare_live_environment()
    workspace = Path(workspace_root).resolve()
    parent_output = Path(parent_output_root).resolve(strict=True)
    output = Path(output_root).resolve()
    dataset_file = Path(dataset_path).resolve(strict=True)
    controller = H23DCampaign(
        LocalTaskDataset(dataset_file).load(),
        dataset=f"local:{dataset_file}",
        workspace_root=workspace,
        output_root=output,
        frozen_output_root=frozen_output_root,
        source_root=source_root,
        model_settings=settings,
        task_timeout_seconds=task_timeout_seconds,
        evaluator_timeout_seconds=evaluator_timeout_seconds,
        repograph_commit=repograph_commit,
        campaign_id=H2_3E_CAMPAIGN_ID,
        stage="H2.3E",
        maximum_live_task_runs=MAX_H2_3E_LIVE_TASK_RUNS,
        planned_base_task_runs=H2_3E_PLANNED_BASE_TASK_RUNS,
        infrastructure_retry_reserve=H2_3E_INFRASTRUCTURE_RETRY_RESERVE,
        configuration_root=parent_output,
        budget_path=workspace / "h2_3e-live-task-runs.jsonl",
        continuation_parent_campaign_id=H2_3D_CAMPAIGN_ID,
    )
    completed_audits = {}
    for name in ("full", "no_correction"):
        experiment = controller.experiments[name]
        report = audit_campaign(
            controller.storage,
            artifacts_root=controller.artifacts_root,
            campaign_id=H2_3D_CAMPAIGN_ID,
            experiment_ids=[experiment.id],
        )
        if not report.passed or report.results != 17:
            raise CampaignError(
                f"H2.3E refuses to continue: parent {name} audit did not pass."
            )
        completed_audits[name] = report.model_dump(mode="json")
    output.mkdir(parents=True, exist_ok=True)
    (output / "parent-audits.json").write_text(
        json.dumps(completed_audits, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    outcome = controller.run()
    combined_audit = audit_campaign(
        controller.storage,
        artifacts_root=controller.artifacts_root,
        campaign_id=H2_3D_CAMPAIGN_ID,
        lineage_campaign_ids=[H2_3E_CAMPAIGN_ID],
    )
    if not combined_audit.passed:
        raise CampaignError("H2.3E combined parent/continuation audit did not pass.")
    (output / "integrity-audit.json").write_text(
        json.dumps(combined_audit.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    authoritative = parent_output / "final"
    _copy_authoritative_view(output, authoritative)
    lineage = {
        "parent_campaign": H2_3D_CAMPAIGN_ID,
        "continuation_campaign": H2_3E_CAMPAIGN_ID,
        "authoritative_view": str(authoritative),
        "completed_parent_configurations_reused": ["full", "no_correction"],
        "continuation_configurations": ["no_exploration", "no_agentic_test"],
    }
    (output / "lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _copy_authoritative_view(output, authoritative)
    scan = scan_persisted_artifacts(
        [workspace, parent_output, output, authoritative]
    )
    security_payload = json.dumps(
        scan.model_dump(mode="json"), indent=2, sort_keys=True
    ) + "\n"
    (output / "security-scan.json").write_text(security_payload, encoding="utf-8")
    (authoritative / "security-scan.json").write_text(
        security_payload, encoding="utf-8"
    )
    return {
        **outcome,
        "parent_campaign": H2_3D_CAMPAIGN_ID,
        "authoritative_view": str(authoritative),
        "parent_audits": completed_audits,
        "combined_audit": combined_audit.model_dump(mode="json"),
    }
