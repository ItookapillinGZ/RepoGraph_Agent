"""Truthful ablation registry over switches the public API really exposes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from evaluation.models import EvaluationConfig


class AblationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    config: EvaluationConfig
    supported: bool = True
    limitation: str | None = None


def ablation_specs() -> list[AblationSpec]:
    """Return four executable configurations over real public switches."""

    return [
        AblationSpec(
            key="A",
            label="Full RepoGraph",
            config=EvaluationConfig(name="full"),
        ),
        AblationSpec(
            key="B",
            label="No Self-Correction",
            config=EvaluationConfig(
                name="no-self-correction",
                self_correct=False,
                max_correction_rounds=0,
            ),
        ),
        AblationSpec(
            key="C",
            label="No Agentic Exploration",
            config=EvaluationConfig(
                name="no-agentic-exploration",
                agentic_explore=False,
                agentic_test=False,
            ),
            limitation=(
                "The existing switch controls candidate-review exploration; "
                "G1 task planning still performs its mandatory bounded exploration."
            ),
        ),
        AblationSpec(
            key="D",
            label="No Agentic Test Tool",
            config=EvaluationConfig(name="no-agentic-test", agentic_test=False),
        ),
    ]
