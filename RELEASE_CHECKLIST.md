# Owner Checklist Before Upload

- Confirm all authors and relevant institutions permit publishing the code.
- Confirm the final paper title and submitted status. No acceptance claim yet.
- Confirm ownership and licensing of paper-specific code, local CityLearn
  changes, parameters, and derived results; add an approved license.
- Keep the upstream CityLearn license and attribution.
- Confirm authorized users can obtain the exact dataset; data is not bundled.
- Resolve the manuscript's five-action formulation versus the implementation's
  three-action set. Do not silently change archived code or experimental results.
- Run `python tools/verify_release.py` and inspect the allowlisted release files.
  Pattern scanning is not a guarantee that every secret has been identified.
- Do not upload the development workspace or its history. This candidate has a
  fresh, empty Git history and excludes original data, logs, virtual environments,
  API secrets, and manuscript drafts.
- Choose GitHub owner, repository name, and visibility. Start private until
  permissions and licensing are confirmed.
- After upload, confirm the actual URL and tag the paper version. Do not use a
  placeholder link in the paper or claim that public hosting already exists.

No GitHub repository has been created, and no remote has been configured by
the local packaging step. The ZIP archive is generated from an explicit file
inventory, excluding local datasets, generated results, caches, and `.git`.
