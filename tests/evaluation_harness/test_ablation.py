"""Executable ablation configuration tests."""

import unittest

from evaluation.experiments.ablation import compare_ablations
from evaluation.experiments.configs import ablation_specs
from tests.evaluation_harness.test_metrics import result


class AblationTests(unittest.TestCase):
    def test_registry_contains_requested_a_to_d(self) -> None:
        specs = ablation_specs()
        self.assertEqual([item.key for item in specs], ["A", "B", "C", "D"])

    def test_test_tool_ablation_is_now_a_real_supported_switch(self) -> None:
        spec = ablation_specs()[-1]
        self.assertTrue(spec.supported)
        self.assertFalse(spec.config.agentic_test)
        self.assertIsNone(spec.limitation)

    def test_supported_configs_map_real_correction_and_exploration_switches(self) -> None:
        full, no_correction, no_explore, _ = ablation_specs()
        self.assertTrue(full.config.self_correct)
        self.assertTrue(full.config.agentic_test)
        self.assertEqual(no_correction.config.effective_correction_rounds, 0)
        self.assertFalse(no_explore.config.agentic_explore)

    def test_comparison_computes_observed_deltas(self) -> None:
        comparison = compare_ablations(
            {
                "Full RepoGraph": [result("one", final=True, tool_calls=4)],
                "No Self-Correction": [result("one", final=False, tool_calls=2)],
            }
        )
        row = comparison.rows[1]
        self.assertEqual(row.delta_resolve_rate_vs_full, -1)
        self.assertEqual(row.delta_tool_calls_vs_full, -2)
        self.assertIn("no statistical-significance", comparison.note)

    def test_comparison_requires_full_baseline(self) -> None:
        with self.assertRaises(ValueError):
            compare_ablations({"Other": []})


if __name__ == "__main__":
    unittest.main()
