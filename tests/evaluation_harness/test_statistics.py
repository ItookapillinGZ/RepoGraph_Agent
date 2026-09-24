"""Deterministic bootstrap and paired comparison tests."""

from __future__ import annotations

import unittest

from evaluation.statistics import bootstrap_paired_delta_ci, bootstrap_resolve_ci


class StatisticsTests(unittest.TestCase):
    def test_same_seed_produces_same_interval(self) -> None:
        first = bootstrap_resolve_ci(
            [True, False, True, False], seed=20260904, resamples=1_000
        )
        second = bootstrap_resolve_ci(
            [True, False, True, False], seed=20260904, resamples=1_000
        )
        self.assertEqual(first, second)

    def test_empty_and_constant_samples_are_truthful(self) -> None:
        empty = bootstrap_resolve_ci([], resamples=10)
        self.assertIsNone(empty.point_estimate)
        self.assertEqual(bootstrap_resolve_ci([True] * 4, resamples=10).lower, 1)
        self.assertEqual(bootstrap_resolve_ci([False] * 4, resamples=10).upper, 0)

    def test_identical_paired_configs_have_zero_delta(self) -> None:
        interval = bootstrap_paired_delta_ci(
            [True, False, True],
            [True, False, True],
            resamples=100,
        )
        self.assertEqual(interval.point_estimate, 0)
        self.assertEqual(interval.lower, 0)
        self.assertEqual(interval.upper, 0)

    def test_paired_samples_require_equal_lengths(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_paired_delta_ci([True], [], resamples=10)


if __name__ == "__main__":
    unittest.main()
