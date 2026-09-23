import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from run_fixed_parameter_control import aggregate_results, extract_trajectory, summarize_periods


class FixedParameterControlTests(unittest.TestCase):
    def trajectory(self):
        return pd.DataFrame({
            "time_step": [0, 1, 2, 3], "cost": [1., 2., 3., 4.],
            "carbon_kg": [2., 4., 6., 8.], "import_kw": [1., 2., 3., 4.],
            "controller_peak_hour": [False, True, False, True],
            "action_value": [0., .1, -.1, 0.], "mean_soc": [.2, .3, .2, .2],
        })

    def test_switch_step_belongs_to_new_stage(self):
        result = summarize_periods(self.trajectory(), [(0, "cost"), (2, "carbon")], "dynamic", 42).set_index("period")
        self.assertEqual(result.loc["stage_1", "cost"], 3)
        self.assertEqual(result.loc["stage_2", "cost"], 7)
        self.assertEqual(result.loc["stage_2", "start_step"], 2)
        self.assertEqual(result.loc["full", "peak_kw"], 4)
        self.assertEqual(result.loc["stage_1", "idle_fraction"], .5)

    def test_current_index_is_not_shifted_or_terminal_tail_included(self):
        env = SimpleNamespace(
            time_step=2, seconds_per_time_step=3600,
            net_electricity_consumption=[5., -2., 999.],
            net_electricity_consumption_cost=[1., 0., 999.],
            net_electricity_consumption_emission=[2., 0., 999.],
            buildings=[SimpleNamespace(electrical_storage=SimpleNamespace(soc=[.2, .3, 0.]))],
        )
        actions = pd.DataFrame({"time_step": [0, 1], "action_value": [.1, -.1],
                                "prior_action": [.1, 0.], "semantic_scenario_key": ["cost", "cost"]})
        trace = extract_trajectory(env, actions)
        np.testing.assert_array_equal(trace.cost, [1., 0.])
        np.testing.assert_array_equal(trace.import_kw, [5., 0.])
        np.testing.assert_array_equal(trace.mean_soc, [.2, .3])

    def test_differences_are_paired_by_seed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for strategy, seed, multiplier in [("dynamic", 42, 1), ("fixed_cost", 42, 2), ("fixed_cost", 43, 10)]:
                run = root / f"seed_{seed}" / strategy
                run.mkdir(parents=True)
                frame = self.trajectory()
                frame["cost"] *= multiplier
                summarize_periods(frame, [(0, "cost")], strategy, seed).to_csv(run / "period_metrics.csv", index=False)
                (run / "complete.json").write_text(json.dumps({"seed": seed}))
            aggregate_results(root)
            pairs = pd.read_csv(root / "paired_differences.csv")
            self.assertEqual(set(pairs.seed), {42})
            self.assertTrue((pairs.cost_delta == -10).all())
            self.assertTrue((pairs.cost_delta_pct == -50).all())


if __name__ == "__main__":
    unittest.main()
