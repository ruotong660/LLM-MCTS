# Local Validation of This Release Candidate

The following checks were performed from the release directory using the
existing Python 3.10 development environment:

- Six unittest tests passed: trajectory alignment, stage boundaries, paired
  differences, archived result arithmetic, basic secret-pattern detection,
  and all-input validation before dataset import/overwrite refusal.
- All 12 archived full-year completion markers contain 8759 controlled steps.
- Annual score arithmetic and paired stage differences match archived results.
- Mean paired low-carbon-stage cost change: +0.304086%; emissions: -0.808046%.
- All 55 historical code/data hashes matched after importing the local dataset.
- Four smoke simulations completed, one for each strategy with seed 42 and
  12 steps. These are execution checks, not additional full-year evidence.
- Figure 2 SVG generation and switch-window statistics generation completed
  using the archived tuned action log and validated input data.
- Release text passed a basic scan for credential patterns and local home paths.
  This is not an exhaustive security audit or a guarantee of no secrets.

Full-year simulations were not rerun for packaging. A clean dependency install
and behavior on other operating systems were not tested. The ZIP excludes
datasets, smoke results, logs, bytecode caches, and Git metadata. Permission
and licensing review remains an owner responsibility before public publication.
