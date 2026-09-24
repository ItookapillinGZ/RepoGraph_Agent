"""Failure taxonomy tests."""

import unittest

from evaluation.failures import FAILURE_CATEGORIES, normalize_failure_category


class FailureTaxonomyTests(unittest.TestCase):
    def test_required_categories_are_present(self) -> None:
        self.assertTrue(
            {
                "planning_failure", "repository_exploration_failure",
                "candidate_generation_failure", "candidate_validation_failure",
                "static_analysis_failure", "test_failure", "review_failure",
                "correction_failure", "timeout", "tool_error",
                "evaluation_test_failure", "infrastructure_error", "unknown",
            }.issubset(FAILURE_CATEGORIES)
        )

    def test_unknown_evidence_is_not_over_inferred(self) -> None:
        self.assertEqual(normalize_failure_category("made-up"), "unknown")
        self.assertEqual(normalize_failure_category(None), "unknown")


if __name__ == "__main__":
    unittest.main()
