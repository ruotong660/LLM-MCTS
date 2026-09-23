"""Matched dynamic/fixed semantic-parameter experiments using the same planner."""

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiment_config import resolve_dataset
import run_llm_cc_mcts as base
import run_llm_cc_mcts_dynamic as dynamic


STRATEGIES = ("dynamic", "fixed_cost", "fixed_carbon", "fixed_peak")
FIXED_SCENARIOS = {
    "fixed_cost": "cost_priority",
    "fixed_carbon": "low_carbon_priority",
    "fixed_peak": "peak_aware",
}
RAW_METRICS = [
    "cost", "carbon_kg", "peak_kw", "controller_peak_kw",
    "mean_action", "idle_fraction", "end_mean_soc",
]
ANNUAL_METRICS = [
    "cost_total", "carbon_emissions_total", "all_time_peak_average",
    "daily_peak_average", "combined_score",
]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_trajectory(env, actions):
    n = len(actions)
    if n == 0 or env.time_step != n:
        raise ValueError("Action log and environment time steps do not match.")
    if not np.array_equal(actions.time_step.to_numpy(), np.arange(n)):
        raise ValueError("Expected a contiguous action log starting at zero.")
    result = actions[["time_step", "action_value", "prior_action",
                      "semantic_scenario_key"]].copy()
    # Runtime updates the current index BEFORE advancing time; do not shift by one.
    for column, attribute in [
        ("grid_kwh", "net_electricity_consumption"),
        ("cost", "net_electricity_consumption_cost"),
        ("carbon_kg", "net_electricity_consumption_emission"),
    ]:
        values = np.asarray(getattr(env, attribute), dtype=float)
        if len(values) < n or not np.isfinite(values[:n]).all():
            raise ValueError(f"Invalid trajectory: {attribute}")
        result[column] = values[:n]
    result["import_kw"] = result.grid_kwh.clip(lower=0) / (env.seconds_per_time_step / 3600)
    result["controller_peak_hour"] = [base.is_peak_hour(t) for t in result.time_step]
    result["mean_soc"] = np.mean([
        np.asarray(b.electrical_storage.soc[:n], dtype=float) for b in env.buildings
    ], axis=0)
    return result


def summarize_periods(trajectory, schedule, strategy, seed):
    n = len(trajectory)
    periods = [("full", "all", 0, n)]
    for i, (start, target) in enumerate(schedule):
        end = schedule[i + 1][0] if i + 1 < len(schedule) else n
        periods.append((f"stage_{i + 1}", target, start, min(end, n)))
    rows = []
    for period, target, start, end in periods:
        frame = trajectory.loc[(trajectory.time_step >= start) & (trajectory.time_step < end)]
        if frame.empty:
            continue
        peak_hours = frame.loc[frame.controller_peak_hour]
        rows.append({
            "strategy": strategy, "seed": seed, "period": period,
            "target": target, "start_step": start, "end_step": end - 1,
            "steps": len(frame), "cost": frame.cost.sum(),
            "carbon_kg": frame.carbon_kg.sum(), "peak_kw": frame.import_kw.max(),
            "controller_peak_kw": peak_hours.import_kw.max(),
            "mean_action": frame.action_value.mean(),
            "idle_fraction": (frame.action_value == 0).mean(),
            "end_mean_soc": frame.mean_soc.iloc[-1],
        })
    return pd.DataFrame(rows)


def aggregate_results(output_dir):
    period_frames, annual_frames = [], []
    for marker in sorted(output_dir.glob("seed_*/*/complete.json")):
        period_frames.append(pd.read_csv(marker.parent / "period_metrics.csv"))
        annual_path = marker.parent / "annual_metrics.csv"
        if annual_path.exists():
            annual_frames.append(pd.read_csv(annual_path))
    if not period_frames:
        return
    periods = pd.concat(period_frames, ignore_index=True)
    periods.to_csv(output_dir / "period_metrics_by_seed.csv", index=False)
    summary = periods.groupby(["strategy", "period", "target"])[RAW_METRICS].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(output_dir / "period_summary.csv")

    dynamic_rows = periods.loc[periods.strategy == "dynamic"]
    fixed_rows = periods.loc[periods.strategy != "dynamic"]
    pairs = fixed_rows.merge(dynamic_rows, on=["seed", "period", "target", "start_step", "end_step", "steps"],
                             suffixes=("_fixed", "_dynamic"), validate="many_to_one")
    differences = pairs[["seed", "period", "target", "strategy_fixed"]].rename(columns={"strategy_fixed": "fixed_strategy"})
    for metric in ["cost", "carbon_kg", "peak_kw", "controller_peak_kw"]:
        difference = pairs[f"{metric}_dynamic"] - pairs[f"{metric}_fixed"]
        differences[f"{metric}_delta"] = difference
        denominator = pairs[f"{metric}_fixed"].abs().replace(0, np.nan)
        differences[f"{metric}_delta_pct"] = 100 * difference / denominator
    differences.to_csv(output_dir / "paired_differences.csv", index=False)
    delta_columns = [c for c in differences if c.endswith("_delta") or c.endswith("_pct")]
    paired_summary = differences.groupby(["fixed_strategy", "period", "target"])[delta_columns].agg(["mean", "std", "count"])
    paired_summary.columns = ["_".join(c) for c in paired_summary.columns]
    paired_summary.to_csv(output_dir / "paired_summary.csv")
    if annual_frames:
        annual = pd.concat(annual_frames, ignore_index=True)
        annual.to_csv(output_dir / "annual_metrics_by_seed.csv", index=False)
        annual_summary = annual.groupby("strategy")[ANNUAL_METRICS].agg(["mean", "std", "count"])
        annual_summary.columns = ["_".join(c) for c in annual_summary.columns]
        annual_summary.to_csv(output_dir / "annual_summary.csv")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cl_2022_p1")
    parser.add_argument("--param-dir", type=Path, default=Path("semantic_params_tune_c_real"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/fixed_parameter_control"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--mcts-iterations", type=int, default=base.MCTS_ITERATIONS)
    parser.add_argument("--rolling-horizon", type=int, default=base.ROLLING_HORIZON)
    parser.add_argument("--exploration-c", type=float, default=base.EXPLORATION_C)
    parser.add_argument("--smoke", action="store_true", help="12-step plumbing check with switches at 4 and 8; not paper evidence.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.mcts_iterations <= 0 or args.rolling_horizon <= 0 or args.exploration_c < 0:
        raise ValueError("Search budgets must be positive and exploration-c nonnegative.")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError("Seeds must lie in [0, 2**32).")
    dataset = resolve_dataset(args.dataset)
    schema = json.loads(dataset.schema_path.read_text())
    if schema.get("simulation_start_time_step", 0) != 0 or schema.get("seconds_per_time_step", 3600) != 3600:
        raise ValueError("This comparison assumes an hourly episode starting at dataset index zero.")
    schedule = dynamic.parse_schedule("0:cost_priority,4:low_carbon_priority,8:peak_aware" if args.smoke else dynamic.DEFAULT_SCHEDULE)
    params = {scenario: base.read_semantic_params(args.param_dir / f"{scenario}.json")
              for scenario in base.SEMANTIC_SCENARIOS}
    mode = "smoke" if args.smoke else "full"
    output_dir = args.result_dir / mode
    sources = [Path(__file__), Path(base.__file__), Path(dynamic.__file__), dataset.schema_path]
    sources += sorted(dataset.data_dir.glob("*.csv"))
    sources += sorted(Path("citylearn").rglob("*.py"))
    config = {
        "dataset": args.dataset, "mode": mode, "schedule": schedule,
        "mcts_iterations": args.mcts_iterations, "rolling_horizon": args.rolling_horizon,
        "exploration_c": args.exploration_c, "action_values": base.ACTION_VALUES,
        "semantic_parameters": params,
        "input_hashes": {str(p): file_hash(p) for p in sources},
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "torch_version": str(base.torch.__version__) if base.torch is not None else None,
    }
    config = json.loads(json.dumps(config))
    manifest = output_dir / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Configuration or source changed. Use a new --result-dir to avoid mixing experiments.")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(config, indent=2, ensure_ascii=True) + "\n")
    snapshot_dir = output_dir / "parameters"
    snapshot_dir.mkdir(exist_ok=True)
    for scenario, values in params.items():
        path = snapshot_dir / f"{scenario}.json"
        path.write_text(json.dumps(values, indent=2, ensure_ascii=True) + "\n")
        base.SEMANTIC_SCENARIOS[scenario]["param_path"] = path
    dynamic.LLM_PROVIDER = None
    base.configure_dataset(dataset.schema_path, dataset.data_dir)
    if not args.smoke and len(base.pricing) <= schedule[-1][0]:
        raise ValueError("Dataset is too short for the prescribed full-year switching schedule.")
    base.EXPLORATION_C = args.exploration_c
    print(f"Results: {output_dir}; completed identical runs are skipped.", flush=True)
    for seed in dict.fromkeys(args.seeds):
        for strategy in dict.fromkeys(args.strategies):
            run_dir = output_dir / f"seed_{seed}" / strategy
            marker = run_dir / "complete.json"
            if marker.exists():
                print(f"Skip completed: seed={seed}, {strategy}", flush=True)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            base.configure_result_dir(run_dir)
            run_schedule = schedule if strategy == "dynamic" else [(0, FIXED_SCENARIOS[strategy])]
            print(f"Running seed={seed}, {strategy}; log: {run_dir / 'run.log'}", flush=True)
            start = time.monotonic()
            with (run_dir / "run.log").open("w") as log, redirect_stdout(log):
                env = dynamic.run_dynamic_experiment(
                    schedule=run_schedule, method_name=strategy,
                    max_real_steps=12 if args.smoke else None,
                    mcts_iterations=args.mcts_iterations, rolling_horizon=args.rolling_horizon,
                    seed=seed,
                )
            actions = pd.read_csv(run_dir / f"{strategy}_actions_debug.csv")
            trajectory = extract_trajectory(env, actions)
            trajectory.to_csv(run_dir / "trajectory.csv", index=False)
            summarize_periods(trajectory, schedule, strategy, seed).to_csv(run_dir / "period_metrics.csv", index=False)
            if not args.smoke:
                if not env.terminated or len(actions) <= schedule[-1][0]:
                    raise RuntimeError("Full run did not cover the prescribed three stages.")
                kpis = pd.read_csv(run_dir / f"{strategy}_kpis.csv")
                district = kpis.loc[kpis.level == "district"].set_index("cost_function").value
                row = {metric: float(district[metric]) for metric in ANNUAL_METRICS if metric != "combined_score"}
                row["combined_score"] = 0.45 * row["cost_total"] + 0.45 * row["carbon_emissions_total"] + 0.10 * row["all_time_peak_average"]
                pd.DataFrame([dict(strategy=strategy, seed=seed, **row)]).to_csv(run_dir / "annual_metrics.csv", index=False)
            elapsed = time.monotonic() - start
            marker.write_text(json.dumps({"seed": seed, "strategy": strategy, "steps": len(actions),
                                          "seconds": elapsed, "smoke_only": args.smoke}, indent=2) + "\n")
            aggregate_results(output_dir)
            print(f"Completed {strategy}: {len(actions)} steps in {elapsed:.1f}s", flush=True)
    aggregate_results(output_dir)
    print("Paired deltas are dynamic minus fixed; negative cost/carbon/peak deltas favor dynamic.")
    if args.smoke:
        print("SMOKE ONLY: shortened stages must not be reported as full-year experimental evidence.")


if __name__ == "__main__":
    main()
