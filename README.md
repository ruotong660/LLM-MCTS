# LLM-MCTS

Research code and archived experimental results for **LLM-Driven Monte Carlo
Tree Search for Dynamic Semantic Optimization in Building Energy Management**.
Submission status: submitted to ICASSP 2027; this repository does not claim acceptance.

This is a curated release candidate, not the entire development workspace.
It contains the locally modified CityLearn implementation used in the experiments.
Do not replace it with a pip-installed CityLearn release when reproducing results.
Public hosting and licensing of the paper-specific contributions are pending
author approval. See [LICENSING.md](LICENSING.md) before publication or reuse.

## Contents

| Path | Purpose |
| --- | --- |
| `run_fixed_parameter_control.py` | Matched dynamic/fixed parameter experiments |
| `run_llm_cc_mcts*.py` | Planner, dynamic schedule, and ablation implementations |
| `run_*baseline.py`, `run_mcts.py`, `run_cc_mcts.py` | Baseline implementations |
| `semantic_params/` | Original parameters for the ablation setting |
| `semantic_params_tune_c_real/` | Tuned parameters for the fixed-parameter comparison |
| `semantic_scenarios/`, `llm_semantic_parser.py` | Natural-language inputs and optional parsing |
| `citylearn/` | Local CityLearn source snapshot, not an unmodified upstream release |
| `artifacts/` | Archived results; never use as a fresh simulation output directory |
| `tools/` | Offline data import and release verification |
| `PROVENANCE.json` | SHA-256 hashes of files copied from the development workspace |

## Quick Start: Inspect Results Without Simulating

From this repository root, using Python 3.10:

```sh
python3 tools/verify_release.py
```

This uses only the Python standard library. It verifies the release inventory,
copied-file hashes, 12 completion markers, and the reported paired percentages.
Archived annual and stage summaries are in `artifacts/fixed_parameter_control/`.

## Install the Simulation Environment

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'
```

`requirements.txt` records direct runtime package versions observed on the
development machine, not a cross-platform lockfile. The release was checked
with the existing development environment; a clean installation on other
platforms has not been validated. `environment-observed.json` records those
versions and Python/platform information. The building-stock generation,
OpenStudio, and visualization workflows outside this paper are not supported
by this minimal dependency list. No `pip install citylearn` is needed.

## Data Setup

Original CityLearn input CSVs are **not redistributed** here. Obtain an
authorized copy of the CityLearn Challenge 2022 Phase 1 dataset, including
`schema.json`, and follow [DATA.md](DATA.md). Import a local dataset folder:

```sh
python tools/import_dataset.py /path/to/citylearn_challenge_2022_phase_1
python tools/verify_release.py --with-data
```

The importer checks all nine input files against hashes saved with the
historical experiments before copying anything. A different schema/data version
must not silently be treated as reproducing these results.

## Run the Fixed-Parameter Comparison

No LLM API key or external LLM request is required for these runs.

```sh
# Four 12-step checks, not scientific results:
python run_fixed_parameter_control.py --smoke --seeds 42

# Twelve full-year runs; potentially many hours:
python run_fixed_parameter_control.py --seeds 42 43 44
```

Outputs go to `results/fixed_parameter_control/`, separate from the archived
`artifacts/`. Completed identical runs are skipped; changed configurations
require a new `--result-dir`. See [FIXED_PARAMETER_CONTROL.md](FIXED_PARAMETER_CONTROL.md).

| Strategy | Steps 0-2999 | Steps 3000-5999 | Steps 6000-8758 |
| --- | --- | --- | --- |
| dynamic | cost | carbon | peak |
| fixed_cost | cost | cost | cost |
| fixed_carbon | carbon | carbon | carbon |
| fixed_peak | peak | peak | peak |

Defaults: 8 search iterations, horizon 4, exploration coefficient 1.2,
shared district action in `{-0.1, 0, 0.1}`. This is the **actual implemented
action set**, not the five-value set appearing in some manuscript drafts.

## What the Results Support

Against fixed cost parameters, low-carbon-stage district emissions decreased
in all three seeds: 0.532%, 0.630%, and 1.262%. Mean paired reduction was
0.808%, with a mean paired cost increase of 0.304%. The peak-aware stage did
not reduce peak demand against this baseline. Fixed carbon parameters had a
slightly better mean annual combined score than dynamic switching.

These are descriptive three-seed results, not a statistical-significance claim.
They test parameter adaptation, **not whether an LLM is necessary**. Episodes
are continuous; prior-stage decisions carry over and no common terminal SOC
constraint was imposed. Stage totals and official normalized annual KPIs use
different aggregations and must not be mixed.

## Paper Tables and Figure

See [EXPERIMENTS.md](EXPERIMENTS.md) for archived-file mappings, limitations,
and commands. The historical main-table score 0.9901 and the three-seed batch
are separate experiments. The main table did not preserve a complete original
source/seed manifest, so exact bitwise reproduction is not promised.

## Optional LLM Parsing

Parser code and original Chinese scenario prompts are included for inspection.
Reproducing the saved-parameter experiments does not require regenerating them.
To make new API calls, set the relevant API key in your environment and pass
an explicitly available model to `parse_semantic_scenarios.py`. Provider/model
metadata in saved JSON files records historical labels, not a guarantee of
current model availability. API calls may incur charges. Never commit keys.

## Before Publishing

Local test results and untested cases are listed in [VALIDATION.md](VALIDATION.md).

Follow [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md). Confirm the paper title,
author approval, licenses, repository owner, and public/private visibility.
No DOI, acceptance status, or GitHub URL has been fabricated for this candidate.
