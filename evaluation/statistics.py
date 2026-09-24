"""Deterministic, dependency-free statistics for small paired benchmarks."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_BOOTSTRAP_SEED = 20260904
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000


class BootstrapInterval(BaseModel):
    """A percentile bootstrap interval with its full analysis provenance."""

    model_config = ConfigDict(extra="forbid")

    point_estimate: float | None
    lower: float | None
    upper: float | None
    confidence_level: float = 0.95
    sample_size: int = Field(ge=0)
    resamples: int = Field(ge=1)
    seed: int


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> BootstrapInterval:
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    observed = [float(value) for value in values]
    if not observed:
        return BootstrapInterval(
            point_estimate=None,
            lower=None,
            upper=None,
            sample_size=0,
            resamples=resamples,
            seed=seed,
        )
    if all(value == observed[0] for value in observed):
        estimate = observed[0]
        return BootstrapInterval(
            point_estimate=estimate,
            lower=estimate,
            upper=estimate,
            sample_size=len(observed),
            resamples=resamples,
            seed=seed,
        )
    # A fixed seed is required for reproducible statistics, not cryptography.
    generator = random.Random(seed)  # nosec B311
    count = len(observed)
    samples = sorted(
        statistics.fmean(observed[generator.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    )
    return BootstrapInterval(
        point_estimate=float(statistics.fmean(observed)),
        lower=_percentile(samples, 0.025),
        upper=_percentile(samples, 0.975),
        sample_size=count,
        resamples=resamples,
        seed=seed,
    )


def bootstrap_resolve_ci(
    resolved: Sequence[bool],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> BootstrapInterval:
    """Return a deterministic 95% CI for the observed resolve rate."""

    return _bootstrap_mean_interval(
        [1.0 if item else 0.0 for item in resolved],
        seed=seed,
        resamples=resamples,
    )


def bootstrap_paired_delta_ci(
    baseline_resolved: Sequence[bool],
    comparison_resolved: Sequence[bool],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> BootstrapInterval:
    """Return a paired CI for baseline minus comparison resolve rate."""

    if len(baseline_resolved) != len(comparison_resolved):
        raise ValueError("paired samples must have identical lengths")
    return _bootstrap_mean_interval(
        [
            float(baseline) - float(comparison)
            for baseline, comparison in zip(
                baseline_resolved,
                comparison_resolved,
                strict=True,
            )
        ],
        seed=seed,
        resamples=resamples,
    )
