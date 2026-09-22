"""One YAML file describes a whole run, and its hash identifies the outputs.

Every stage takes ``--config``. Nothing takes a bare threshold on the command
line, because a number typed at a shell prompt is not reproducible and the v0
pipeline's fifteen near-identical per-round configs are what happens when the
alternative is to copy a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import yaml

from .gates import Gates
from .gesture import GestureParams
from .qualify import Qualifiers
from .scan import ScanSettings

DEFAULT_CONFIG = Path("configs/vibes_upper_body.yaml")


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    source_root: Path
    model_root: Path
    inventory: Path
    output_root: Path
    private_root: Path
    vendors: tuple[str, ...]
    scan: ScanSettings
    scan_tasks: int
    gates: Gates
    qualifiers: Qualifiers
    max_clips_per_file: int
    max_files_per_participant: int
    select_seed: str
    clip_seconds: float
    render_tasks: int
    raw: Mapping[str, Any] = field(repr=False, default_factory=dict)

    # ---- derived paths ----------------------------------------------------
    @property
    def population_path(self) -> Path:
        return self.output_root / "population.parquet"

    @property
    def shard_dir(self) -> Path:
        return self.output_root / "scan_shards"

    @property
    def windows_path(self) -> Path:
        return self.output_root / "windows.parquet"

    @property
    def files_path(self) -> Path:
        return self.output_root / "scan_files.parquet"

    @property
    def candidates_path(self) -> Path:
        return self.output_root / "candidates.parquet"

    @property
    def qualified_path(self) -> Path:
        """Every candidate clip with its tier-2 scores, flags and verdict."""

        return self.output_root / "qualified_clips.parquet"

    @property
    def qualification_funnel_path(self) -> Path:
        return self.output_root / "qualification_funnel.csv"

    @property
    def reviewed_clips_path(self) -> Path:
        """The manually reviewed subset. A development artefact, not production."""

        return self.output_root / "reviewed_clips.csv"

    @property
    def review_manifest_path(self) -> Path:
        return self.output_root / "review_manifest.csv"

    @property
    def verdict_log(self) -> Path:
        return self.output_root / "review_verdicts.jsonl"

    @property
    def media_root(self) -> Path:
        return self.private_root / "clips"

    @property
    def accepted_clips_path(self) -> Path:
        return self.output_root / "accepted_clips.csv"

    @property
    def accepted_segments_path(self) -> Path:
        return self.output_root / "accepted_segments.csv"

    def config_hash(self) -> str:
        return sha256(yaml.safe_dump(dict(self.raw), sort_keys=True).encode()).hexdigest()[:16]


def load_config(path: str | Path = DEFAULT_CONFIG) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: top level must be a mapping")
    if int(raw.get("schema_version", 0)) != 2:
        raise ValueError(f"{path}: expected schema_version 2")

    outputs = dict(raw.get("outputs") or {})
    scan_block = dict(raw.get("scan") or {})
    gesture_block = dict(scan_block.pop("gesture", None) or {})
    select_block = dict(raw.get("select") or {})
    render_block = dict(raw.get("render") or {})
    gate_block = dict(raw.get("gates") or {})
    qualify_block = dict(raw.get("qualify") or {})

    unknown = set(gate_block) - set(Gates().as_dict())
    if unknown:
        raise ValueError(f"{path}: unknown gate keys {sorted(unknown)}")
    unknown = set(gesture_block) - set(GestureParams().__dataclass_fields__)
    if unknown:
        raise ValueError(f"{path}: unknown gesture keys {sorted(unknown)}")

    unknown = set(qualify_block) - set(Qualifiers().as_dict())
    if unknown:
        raise ValueError(f"{path}: unknown qualify keys {sorted(unknown)}")

    gates = replace(Gates(), **gate_block) if gate_block else Gates()
    qualifiers = replace(Qualifiers(), **qualify_block) if qualify_block else Qualifiers()
    gesture = replace(GestureParams(), **gesture_block) if gesture_block else GestureParams()

    return RunConfig(
        run_id=str(raw["run_id"]),
        source_root=Path(raw["source_root"]),
        model_root=Path(raw.get("model_root", "model_files")),
        inventory=Path(raw["inventory"]),
        output_root=Path(outputs["root"]),
        private_root=Path(outputs["private_root"]),
        vendors=tuple(raw.get("vendors") or ("V00", "V01", "V02", "V03")),
        scan=ScanSettings(
            window_seconds=float(scan_block.get("window_seconds", 30.0)),
            hop_seconds=float(scan_block.get("hop_seconds", 10.0)),
            model_root=str(raw.get("model_root", "model_files")),
            gesture=gesture,
        ),
        scan_tasks=int(scan_block.get("tasks", 512)),
        gates=gates,
        qualifiers=qualifiers,
        max_clips_per_file=int(select_block.get("max_clips_per_file", 8)),
        max_files_per_participant=int(select_block.get("max_files_per_participant", 12)),
        select_seed=str(select_block.get("seed", "vibes")),
        clip_seconds=float(render_block.get("clip_seconds", 30.0)),
        render_tasks=int(render_block.get("tasks", 256)),
        raw=raw,
    )
