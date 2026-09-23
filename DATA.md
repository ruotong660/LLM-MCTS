# Data Access and Validation

Dataset: CityLearn Challenge 2022 Phase 1 (`cl_2022_p1`). Start with the
CityLearn project's dataset resources and the dataset provider's usage terms:
https://github.com/intelligent-environments-lab/CityLearn

This release does not download or redistribute input data automatically.
The author already has a local copy in the development workspace. Public users
must independently obtain an authorized copy with matching input files.

The nine required files are `schema.json`, `Building_1.csv` through
`Building_5.csv`, `pricing.csv`, `carbon_intensity.csv`, and `weather.csv`.
Their expected SHA-256 hashes are preserved in
`artifacts/fixed_parameter_control/config.json`, under `input_hashes`.

```sh
python tools/import_dataset.py /path/to/citylearn_challenge_2022_phase_1
```

All source files are checked before any copy. Existing identical destination
files are accepted; different files are not overwritten. Files are placed in
`data/datasets/citylearn_challenge_2022_phase_1/`, which Git ignores.
If any hash differs, investigate the dataset/schema version rather than
disabling the check. Matching external data availability has not been verified.

The schema has 8760 hourly states; the saved experiments execute 8759 actions.
The final state must not be mistaken for an additional controlled transition.
The supported publication workflow is limited to CL-2022-P1; other datasets
from the original development workspace are intentionally omitted.
