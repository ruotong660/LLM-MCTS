# Experiment-to-Artifact Map

## Historical Main Table

The seven KPI files in `artifacts/main_table/` correspond to:

| Paper method | KPI filename |
| --- | --- |
| No Control | `no_control_kpis.csv` |
| RBC | `official_rbc_kpis.csv` |
| Rolling MPC | `rolling_mpc_kpis.csv` |
| SAC | `sac_baseline_kpis.csv` |
| MCTS | `standard_mcts_kpis.csv` |
| CC-MCTS | `cc_mcts_kpis.csv` |
| LLM-MCTS | `llm_cc_mcts_dynamic_tune_c_real_kpis.csv` |

Use district-level rows. Combined score is 0.45 * normalized cost +
0.45 * normalized carbon + 0.10 * normalized all-time peak. The main-table
LLM-MCTS result is approximately 0.9901, distinct from the paired-seed batch.
Historical baselines do not have a complete archived seed/source manifest;
the current scripts are provided, not certified as exact historical snapshots.
Some legacy baseline scripts run a full simulation without CLI options.
Inspect them before executing; do not assume `--help` prevents execution.

## Original-Parameter Ablations

`artifacts/ablation/` contains the five KPI files and the original summary.
The full dynamic reference has combined score approximately 0.9935.
Original parameters in `semantic_params/` differ from the tuned directory.
Run each ablation in a separate process (the script patches planner functions):

```sh
python run_llm_cc_mcts_dynamic.py --dataset cl_2022_p1 --result-dir results/ablation
python run_llm_cc_mcts_dynamic_ablation.py --dataset cl_2022_p1 --ablation no_prior --result-dir results/ablation
python run_llm_cc_mcts_dynamic_ablation.py --dataset cl_2022_p1 --ablation no_action_sort --result-dir results/ablation
python run_llm_cc_mcts_dynamic_ablation.py --dataset cl_2022_p1 --ablation no_peak_penalty --result-dir results/ablation
python run_llm_cc_mcts_dynamic_ablation.py --dataset cl_2022_p1 --ablation no_smooth --result-dir results/ablation
```

These are fresh runs, not replacements for the archived results. The
`no_prior` implementation also disables prior-based action ordering; interpret
it as removal of prior guidance, not a perfectly isolated single scalar term.

## Figure 2

The saved tuned action log is in `artifacts/figure2/`. With input data installed:

```sh
python plot_dynamic_switch_analysis.py --actions-path artifacts/figure2/llm_cc_mcts_dynamic_tune_c_real_actions_debug.csv --output-dir results/figures
```

This produces SVG plots and window statistics. The plot is a local descriptive
view around switches, not a causal comparison of control strategies. Some
parameter JSON explanatory strings predate numerical tuning; actual numerical
fields, not the free-text explanations, determine the planner's behavior.

## Paired Fixed-Parameter Study

`artifacts/fixed_parameter_control/` preserves the historical configuration,
parameter snapshots, six aggregate CSVs, and per-run completion/annual/stage
metrics for all twelve runs. Full trajectories and machine-specific logs are
not distributed in this compact candidate. Fresh runs regenerate them.

Absolute file-path keys in the historical configuration were normalized to
workspace-relative paths for privacy; all input hash values are unchanged.
The manifest's source hashes match the copied core planner and CityLearn
sources. Seeds are 42, 43, 44; defaults are 8 iterations and horizon 4. The
saved parameters are reused without new LLM calls. `tools/verify_release.py`
recomputes paired stage changes from the per-run period metrics and checks
the archived annual score formula. See `FIXED_PARAMETER_CONTROL.md` for units,
stage boundaries, SOC carryover, and reproducibility caveats.
