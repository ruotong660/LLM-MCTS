"""Import an authorized local dataset only if it matches the experiment hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "data/datasets/citylearn_challenge_2022_phase_1/"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_dataset(source):
    config = json.loads((ROOT / "artifacts/fixed_parameter_control/config.json").read_text())
    entries = [(ROOT / name, source / Path(name).name, expected)
               for name, expected in config["input_hashes"].items()
               if name.startswith(PREFIX)]
    if len(entries) != 9:
        raise ValueError("Expected the nine historical input files.")
    for dest, src, expected in entries:
        if not src.is_file() or digest(src) != expected:
            raise ValueError(f"Missing or mismatched source: {src.name}; no files copied.")
        if dest.exists() and digest(dest) != expected:
            raise ValueError(f"Refusing to overwrite different destination: {dest.name}")
    for dest, src, _ in entries:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(src, dest)
    print(f"Validated and imported {len(entries)} files. Dataset remains Git-ignored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    import_dataset(args.source)
