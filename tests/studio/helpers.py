"""Deterministic Studio test fixtures."""

from __future__ import annotations

import atexit
import os
import tempfile
from pathlib import Path

from engineering_plan import EngineeringPlan, PlannedFileChange
from plan_application import PlanApplicationBundle
from plan_execution import MultiFileCandidate

_boot_directory = tempfile.TemporaryDirectory(prefix="repograph-studio-tests-")
atexit.register(_boot_directory.cleanup)
BOOT_ROOT = Path(_boot_directory.name)
BOOT_WORKSPACE = BOOT_ROOT / "workspace"
BOOT_DATA = BOOT_ROOT / "data"
BOOT_WORKSPACE.mkdir()
BOOT_DATA.mkdir()
os.environ.setdefault("REPOGRAPH_WORKSPACE_ROOT", str(BOOT_WORKSPACE))
os.environ.setdefault("REPOGRAPH_STUDIO_DATA_DIR", str(BOOT_DATA))
os.environ.setdefault("REPOGRAPH_STUDIO_ALLOWED_ORIGIN", "http://localhost:3000")


DIGEST = "d" * 64


def application_bundle() -> PlanApplicationBundle:
    plan = EngineeringPlan(
        summary="Update one file safely.",
        files=[
            PlannedFileChange(
                path="sample.py",
                action="modify",
                rationale="Cover the requested behavior.",
            )
        ],
    )
    candidate = MultiFileCandidate(
        summary="Updated implementation.",
        files=[{"path": "sample.py", "action": "modify", "content": "value = 2\n"}],
    )
    return PlanApplicationBundle(
        plan=plan,
        candidate=candidate,
        diff_text="--- a/sample.py\n+++ b/sample.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n",
        expected_original_hashes={"sample.py": "a" * 64},
        expected_candidate_hashes={"sample.py": "b" * 64},
        approval_digest=DIGEST,
    )
