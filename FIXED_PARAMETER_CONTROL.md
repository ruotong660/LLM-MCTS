# Fixed-Parameter Control Experiment

## Question

Does updating structured semantic parameters at prescribed objective switches
change scheduling performance compared with keeping the parameters fixed?
This experiment tests parameter adaptation, not whether an LLM is better than
human-written parameter mappings. No LLM API requests or new tuning are needed.

## Matched Conditions

| Strategy | Steps 0-2999 | Steps 3000-5999 | Steps 6000-end |
| --- | --- | --- | --- |
| dynamic | cost_priority | low_carbon_priority | peak_aware |
| fixed_cost | cost_priority | cost_priority | cost_priority |
| fixed_carbon | low_carbon_priority | low_carbon_priority | low_carbon_priority |
| fixed_peak | peak_aware | peak_aware | peak_aware |

All strategies use the same planner, dataset, initial seed, action set,
iterations, horizon, and exploration coefficient. Fixed parameters do NOT mean
fixed actions: the prior and selected action still respond to numerical inputs.
Every strategy runs a continuous episode, without resetting SOC at switches.
Later stage differences therefore include the effects of earlier decisions.

The default parameter source is `semantic_params_tune_c_real/`, not
`semantic_params/`. Existing static result files use different settings and
must not be substituted into this matched comparison. All four methods are
rerun, including the dynamic reference. An exact reproduction of the historical
0.9901 score is not assumed: the historical run did not save a complete source
and random-seed manifest.

Current defaults: CL-2022-P1, 8 MCTS iterations, horizon 4, exploration 1.2,
shared district action in {-0.1, 0, 0.1}. These are implementation settings;
check that the submitted paper describes the actual experiment settings.

## Run From the Repository Root

Check the four execution paths first (switches at 4 and 8, only 12 steps):

```sh
.venv/bin/python run_fixed_parameter_control.py --smoke --seeds 42
```

Run all four full-year conditions with one paired seed:

```sh
.venv/bin/python run_fixed_parameter_control.py --seeds 42
```

Then complete a predetermined set of seeds for descriptive mean/std reporting:

```sh
.venv/bin/python run_fixed_parameter_control.py --seeds 42 43 44
```

The last command skips already completed identical runs. Do not select seeds
based on favorable results. Three seeds are an initial variability check, not
strong evidence of statistical significance.

Each seed has four full-year simulations. Historical dynamic logs averaged
about 0.94 seconds per decision (8759 decisions), suggesting roughly 9 hours
for four runs, but actual runtime depends on the machine and code version.
The smoke test validates execution; it does not estimate late-year deepcopy
cost reliably and must not be reported as full-year evidence.

## Outputs

Full runs go to `results/fixed_parameter_control/full/`; smoke runs have their
own `smoke/` directory. Original experiment results are not overwritten.

- `config.json`: source/data hashes, parameter values, and search settings.
- `parameters/`: the actual JSON parameter snapshots used by every condition.
- `seed_42/dynamic/run.log`: progress and planner settings for an individual run.
- `seed_42/dynamic/trajectory.csv`: actual district cost, emissions, grid energy,
  actions, and mean SOC for controlled steps (index zero is the first action).
- `annual_metrics_by_seed.csv`: official normalized annual KPIs and combined score.
- `annual_summary.csv`: per-strategy mean, sample std, and count.
- `period_metrics_by_seed.csv`: full-run and stage-wise physical metrics.
- `paired_differences.csv`: dynamic minus fixed, paired by seed and stage.
- `paired_summary.csv`: descriptive statistics of the paired differences.

An interrupted unfinished condition reruns from its beginning; completed
conditions are skipped. Changed code, data, or parameters require a new
`--result-dir` to prevent mixing incompatible runs. Only completion-marked
conditions enter summary files. While a batch is incomplete, inspect counts:
per-strategy summaries can contain different seed sets; paired differences
contain only matching seed/stage pairs.

## Interpretation and Paper Table

Use `fixed_cost` as the direct "do not update after initialization" control.
Report `fixed_carbon` and `fixed_peak` as additional static alternatives,
including unfavorable results. Do not choose only the easiest fixed baseline.

Compare the SAME stage across strategies, not one stage against another:

- Stage 1: operating cost (`cost`). Dynamic and fixed_cost should match for
  each seed before the first switch; this is a useful reproducibility check.
- Stage 2: actual carbon emissions (`carbon_kg`).
- Stage 3: district import peak (`peak_kw`), with `controller_peak_kw` as a
  secondary diagnostic restricted to the planner's peak-hour index rule
  `17 <= time_step % 24 <= 21`. Do not relabel it as local clock time without
  checking the dataset's hour convention.
- All stages: inspect `end_mean_soc`; reduced grid use can reflect different
  stored-energy levels. This protocol has no added common terminal SOC constraint.
- Full year: retain official cost/carbon/all-time-peak KPIs and combined score
  `0.45 * cost_total + 0.45 * carbon_emissions_total + 0.10 * all_time_peak_average`.

Stage cost/emissions are unnormalized district sums in dataset currency/kg CO2.
Stage peak is max(max(district net kWh, 0))/step_hours. They are not the same
aggregation as every official normalized annual CityLearn KPI. Avoid mixing
these units or calling the heuristic action prior score a realized cost.

A compact paper table can have four rows (the strategies) and columns:
Stage-1 cost, Stage-2 CO2, Stage-3 peak kW, annual combined score. Report
mean +/- sample std with the number of seeds; a single-seed std is undefined.
Negative paired cost/carbon/peak deltas favor dynamic; positive deltas favor
fixed. Show the results even if dynamic is not the best in every column.

Interpretation should follow the evidence. Consistent target-metric gains
support usefulness of parameter adaptation under this schedule. Small or
mixed differences support a narrower claim about changed preferences and
competitive performance. Neither outcome alone establishes LLM necessity.

Suggested setup paragraph (add results only after full runs complete):

```latex
To isolate the effect of parameter updates, we compare dynamic switching
with three controls that retain the cost-priority, low-carbon, or peak-aware
parameter setting throughout the episode. All conditions use identical
parameter sources, environments, search budgets, and paired random seeds.
We compare realized cost, carbon emissions, and import peaks over the same
three stages and report the annual combined score. Each run is continuous,
so stage-wise differences include the effects of preceding decisions.
```
