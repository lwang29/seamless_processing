#!/usr/bin/env python3
"""Create the deterministic, vendor-stratified Session 1 dev manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import yaml


def _allocate_proportionally(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder proportional allocation, with one item per stratum."""
    if total < len(counts):
        raise ValueError("sample is too small to include every vendor stratum")
    available = sum(counts.values())
    raw = {key: total * value / available for key, value in counts.items()}
    allocation = {key: max(1, int(value)) for key, value in raw.items()}

    while sum(allocation.values()) > total:
        candidates = [key for key, value in allocation.items() if value > 1]
        key = min(candidates, key=lambda item: raw[item] - allocation[item])
        allocation[key] -= 1
    while sum(allocation.values()) < total:
        candidates = [key for key in counts if allocation[key] < counts[key]]
        key = max(candidates, key=lambda item: raw[item] - allocation[item])
        allocation[key] += 1
    return allocation


def build_manifest(config_path: Path) -> tuple[list[dict[str, str]], Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_root = Path(config["source_root"]).resolve()
    split = str(config["split"])
    seed = int(config["sample"]["seed"])
    files_per_label = int(config["sample"]["files_per_label"])
    output = Path(config["sample"]["manifest"])

    rows: list[dict[str, str]] = []
    for label in config["labels"]:
        strata: dict[str, list[Path]] = defaultdict(list)
        for npz_path in sorted((source_root / label / split).glob("*/*/*.npz")):
            vendor = npz_path.stem.split("_", 1)[0]
            strata[vendor].append(npz_path)
        if not strata:
            raise FileNotFoundError(f"no NPZ files found for {label}/{split}")

        allocation = _allocate_proportionally(
            {vendor: len(paths) for vendor, paths in strata.items()}, files_per_label
        )
        for vendor in sorted(strata):
            stratum_seed = hashlib.sha256(
                f"{seed}:{label}:{vendor}".encode("utf-8")
            ).digest()
            rng = random.Random(int.from_bytes(stratum_seed[:8], "big"))
            chosen = sorted(rng.sample(strata[vendor], allocation[vendor]))
            for npz_path in chosen:
                base_path = npz_path.with_suffix("")
                relative_base = base_path.relative_to(source_root)
                rows.append(
                    {
                        "file_id": base_path.name,
                        "label": str(label),
                        "split": split,
                        "vendor": vendor,
                        "shard_group": relative_base.parts[2],
                        "shard_id": relative_base.parts[3],
                        "source_relbase": relative_base.as_posix(),
                    }
                )

    rows.sort(key=lambda row: (row["label"], row["vendor"], row["file_id"]))
    return rows, output


def build_dyad_manifest(config_path: Path) -> tuple[list[dict[str, str]], Path]:
    """Select complete dyads for frame-count and time-origin checks."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_root = Path(config["source_root"]).resolve()
    split = str(config["split"])
    seed = int(config["dyad_audit"]["seed"])
    pairs_per_vendor = int(config["dyad_audit"]["pairs_per_vendor"])
    output = Path(config["dyad_audit"]["manifest"])

    rows: list[dict[str, str]] = []
    for label in config["labels"]:
        interactions: dict[str, list[Path]] = defaultdict(list)
        for npz_path in sorted((source_root / label / split).glob("*/*/*.npz")):
            interaction_key = "_".join(npz_path.stem.split("_")[:3])
            interactions[interaction_key].append(npz_path)

        complete_by_vendor: dict[str, list[tuple[str, list[Path]]]] = defaultdict(list)
        for interaction_key, paths in interactions.items():
            if len(paths) == 2:
                complete_by_vendor[interaction_key.split("_", 1)[0]].append(
                    (interaction_key, sorted(paths))
                )

        for vendor in sorted(complete_by_vendor):
            choices = sorted(complete_by_vendor[vendor])
            if len(choices) < pairs_per_vendor:
                raise ValueError(f"not enough complete dyads in {label}/{vendor}")
            stratum_seed = hashlib.sha256(
                f"{seed}:{label}:{vendor}".encode("utf-8")
            ).digest()
            rng = random.Random(int.from_bytes(stratum_seed[:8], "big"))
            for interaction_key, paths in sorted(rng.sample(choices, pairs_per_vendor)):
                for member_index, npz_path in enumerate(paths):
                    relative_base = npz_path.with_suffix("").relative_to(source_root)
                    rows.append(
                        {
                            "interaction_key": interaction_key,
                            "member_index": str(member_index),
                            "file_id": npz_path.stem,
                            "label": str(label),
                            "split": split,
                            "vendor": vendor,
                            "source_relbase": relative_base.as_posix(),
                        }
                    )
    rows.sort(
        key=lambda row: (
            row["label"],
            row["vendor"],
            row["interaction_key"],
            row["member_index"],
        )
    )
    return rows, output


def build_movement_manifest(config_path: Path) -> tuple[list[dict[str, str]], Path]:
    """Supplement the union to 100 V00 files, where movement:is_valid exists."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_root = Path(config["source_root"]).resolve()
    split = str(config["split"])
    seed = int(config["movement_audit"]["seed"])
    target_per_label = int(config["movement_audit"]["target_v00_files_per_label"])
    output = Path(config["movement_audit"]["manifest"])

    main_rows, _ = build_manifest(config_path)
    dyad_rows, _ = build_dyad_manifest(config_path)
    already_selected = {row["file_id"] for row in main_rows + dyad_rows}
    rows: list[dict[str, str]] = []
    for label in config["labels"]:
        all_v00 = sorted(
            path
            for path in (source_root / label / split).glob("*/*/V00_*.npz")
        )
        existing = [path for path in all_v00 if path.stem in already_selected]
        needed = target_per_label - len(existing)
        if needed < 0:
            raise ValueError(f"existing {label} V00 sample exceeds requested target")
        candidates = [path for path in all_v00 if path.stem not in already_selected]
        stratum_seed = hashlib.sha256(f"{seed}:{label}:V00".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(stratum_seed[:8], "big"))
        for npz_path in sorted(rng.sample(candidates, needed)):
            relative_base = npz_path.with_suffix("").relative_to(source_root)
            rows.append(
                {
                    "file_id": npz_path.stem,
                    "label": str(label),
                    "split": split,
                    "vendor": "V00",
                    "source_relbase": relative_base.as_posix(),
                }
            )
    rows.sort(key=lambda row: (row["label"], row["file_id"]))
    return rows, output


def _write_csv_atomic(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/recon.yaml"))
    args = parser.parse_args()
    rows, output = build_manifest(args.config)
    _write_csv_atomic(rows, output)
    print(f"wrote {len(rows)} rows to {output}")
    dyad_rows, dyad_output = build_dyad_manifest(args.config)
    _write_csv_atomic(dyad_rows, dyad_output)
    print(f"wrote {len(dyad_rows)} rows to {dyad_output}")
    movement_rows, movement_output = build_movement_manifest(args.config)
    _write_csv_atomic(movement_rows, movement_output)
    print(f"wrote {len(movement_rows)} rows to {movement_output}")


if __name__ == "__main__":
    main()
