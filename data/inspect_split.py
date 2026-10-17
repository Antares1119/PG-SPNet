"""Export Reynolds-number case groups from supplied HDF5 split files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

def case_keys(path: Path) -> list[float]:
    with h5py.File(path, "r") as stream:
        return sorted(float(key) for key in stream.keys())

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect train, validation and test Re groups")
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = {name: case_keys(getattr(args, name)) for name in ("train", "validation", "test")}
    sets = {name: set(values) for name, values in groups.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise ValueError(f"Overlapping cases between {left} and {right}: {sorted(overlap)}")
    args.output.write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    print({name: len(values) for name, values in groups.items()})

if __name__ == "__main__":
    main()
