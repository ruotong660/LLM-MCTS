"""Offline integrity, result arithmetic, and basic sensitive-file checks."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("dynamic", "fixed_cost", "fixed_carbon", "fixed_peak")
PATTERNS = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "provider token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "github token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "local home path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "literal credential": re.compile(
        r'''(?i)(?:api_key|password|access_token)\s*[=:]\s*["'][A-Za-z0-9_./+-]{16,}["']'''
    ),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def scan_text(text):
    return [name for name, pattern in PATTERNS.items() if pattern.search(text)]


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify_inventory(root):
    inventory = json.loads((root / "RELEASE_FILES.json").read_text())
    for relative, expected in inventory.items():
        path = root / relative
        check(path.is_file() and not path.is_symlink(), f"Missing/linked file: {relative}")
        check(sha256(path) == expected, f"Changed release file: {relative}")
        issues = scan_text(path.read_text(encoding="utf-8"))
        check(not issues, f"Sensitive pattern in {relative}: {', '.join(issues)}")
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in {".git", ".venv", "__pycache__", ".pytest_cache"} for part in rel.parts):
            continue
        if rel.parts[0] == "results" or rel.as_posix().startswith("data/datasets/"):
            continue
        if path.is_file():
            check(rel.as_posix() in inventory or rel.name == "RELEASE_FILES.json",
                  f"Unlisted file: {rel}")
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    for relative, record in provenance["copied_files"].items():
        check(sha256(root / relative) == record["sha256"], f"Provenance mismatch: {relative}")
    print(f"PASS: {len(inventory)} inventoried files; copied-file hashes and pattern scan.")


def verify_results(root):
    base = root / "artifacts/fixed_parameter_control"
    annual = rows(base / "annual_metrics_by_seed.csv")
    expected = {(str(seed), strategy) for seed in (42, 43, 44) for strategy in STRATEGIES}
    check(len(annual) == 12 and {(r["seed"], r["strategy"]) for r in annual} == expected,
          "Expected exactly twelve paired annual runs.")
    periods = {}
    for seed, strategy in sorted(expected):
        run = base / f"seed_{seed}" / strategy
        marker = json.loads((run / "complete.json").read_text())
        check(marker["steps"] == 8759 and marker["smoke_only"] is False
              and str(marker["seed"]) == seed and marker["strategy"] == strategy,
              f"Invalid completion marker: {seed}/{strategy}")
        records = rows(run / "period_metrics.csv")
        check(len(records) == 4, "Expected full year and three stages.")
        for r in records:
            check(r["seed"] == seed and r["strategy"] == strategy, "Mislabelled period row.")
            for key in ("cost", "carbon_kg", "peak_kw"):
                check(math.isfinite(float(r[key])), f"Nonfinite metric: {key}")
            periods[seed, strategy, r["period"]] = r
        for period, start, end in (("stage_1", 0, 2999), ("stage_2", 3000, 5999),
                                   ("stage_3", 6000, 8758), ("full", 0, 8758)):
            r = periods[seed, strategy, period]
            check((int(r["start_step"]), int(r["end_step"]), int(r["steps"])) ==
                  (start, end, end - start + 1), "Incorrect stage boundary.")
    for r in annual:
        score = .45 * float(r["cost_total"]) + .45 * float(r["carbon_emissions_total"]) + .10 * float(r["all_time_peak_average"])
        check(math.isclose(score, float(r["combined_score"]), abs_tol=1e-12), "Annual score mismatch.")
        per_run = rows(base / f"seed_{r['seed']}" / r["strategy"] / "annual_metrics.csv")
        check(len(per_run) == 1 and per_run[0].keys() == r.keys(), "Annual archive schema mismatch.")
        for key, value in r.items():
            equal = (per_run[0][key] == value if key in {"seed", "strategy"} else
                     math.isclose(float(per_run[0][key]), float(value), rel_tol=0, abs_tol=1e-12))
            check(equal, f"Annual archive differs from per-run result: {key}")
    for pair in rows(base / "paired_differences.csv"):
        a = periods[pair["seed"], "dynamic", pair["period"]]
        b = periods[pair["seed"], pair["fixed_strategy"], pair["period"]]
        for metric in ("cost", "carbon_kg", "peak_kw", "controller_peak_kw"):
            delta = float(a[metric]) - float(b[metric])
            check(math.isclose(delta, float(pair[f"{metric}_delta"]), abs_tol=1e-9), "Paired delta mismatch.")
            pct = 100 * delta / abs(float(b[metric]))
            check(math.isclose(pct, float(pair[f"{metric}_delta_pct"]), abs_tol=1e-9), "Paired percentage mismatch.")
    costs, carbon = [], []
    for seed in ("42", "43", "44"):
        a = periods[seed, "dynamic", "stage_2"]
        b = periods[seed, "fixed_cost", "stage_2"]
        costs.append(100 * (float(a["cost"]) / float(b["cost"]) - 1))
        carbon.append(100 * (float(a["carbon_kg"]) / float(b["carbon_kg"]) - 1))
        for metric in ("cost", "carbon_kg", "peak_kw"):
            check(periods[seed, "dynamic", "stage_1"][metric] ==
                  periods[seed, "fixed_cost", "stage_1"][metric], "Initial stage mismatch.")
    check(all(x < 0 for x in carbon), "Expected carbon reductions in all three seeds.")
    check(round(statistics.mean(costs), 3) == .304 and round(statistics.mean(carbon), 3) == -.808,
          "Reported rounded percentages do not match.")
    print(f"PASS: 12 complete runs; mean paired cost {statistics.mean(costs):+.6f}%, carbon {statistics.mean(carbon):+.6f}%.")


def verify_historical_inputs(root, with_data):
    config = json.loads((root / "artifacts/fixed_parameter_control/config.json").read_text())
    checked = 0
    for relative, expected in config["input_hashes"].items():
        if relative.startswith("data/") and not with_data:
            continue
        path = root / relative
        check(path.is_file() and sha256(path) == expected, f"Historical input mismatch: {relative}")
        checked += 1
    print(f"PASS: {checked} historical source/data hashes; data {'included' if with_data else 'not checked (not bundled)' }.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-data", action="store_true")
    args = parser.parse_args()
    verify_inventory(ROOT)
    verify_results(ROOT)
    verify_historical_inputs(ROOT, args.with_data)
