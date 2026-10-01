"""One YAML file describes a whole annotation run, and its hash identifies the outputs.

Every stage takes ``--config``. Nothing takes a bare threshold on the command
line, because a number typed at a shell prompt is not reproducible.

The run is split by cost. ``scan`` and ``scan-video`` read the release (hours of
cluster time) and store **continuous measurements only**; every threshold,
label, flag and normalisation lives in ``annotate`` (minutes of pandas). So the
``posture:`` block and the clip length's downstream meaning can be re-tuned
without touching the release again — only ``clips.seconds`` and the ``scan``
block change what the scan measures, and both are hashed into its fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import yaml

from .gesture import GestureParams

DEFAULT_CONFIG = Path("configs/annotations_v1.yaml")
SCHEMA_VERSION = 3


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    source_root: Path
    model_root: Path
    inventory: Path
    metadata_root: Path
    output_root: Path
    clip_seconds: float
    scan_tasks: int
    gesture: GestureParams
    video_short_side: int
    posture_overrides: Mapping[str, Any]
    legacy_run_root: Path | None
    raw: Mapping[str, Any] = field(repr=False, default_factory=dict)

    # ---- derived paths ----------------------------------------------------
    @property
    def catalog_dir(self) -> Path:
        return self.output_root / "catalog"

    @property
    def scan_dir(self) -> Path:
        return self.output_root / "scan_shards"

    @property
    def video_dir(self) -> Path:
        return self.output_root / "video_shards"

    @property
    def annotations_dir(self) -> Path:
        """The published tables. Written as a whole, atomically."""
        return self.output_root / "annotations"

    @property
    def legacy_dir(self) -> Path:
        """Labels carried over from the previous (filtering) iteration."""
        return self.output_root / "legacy"

    @property
    def qa_dir(self) -> Path:
        return self.output_root / "qa"

    def posture_rules(self):
        from .posture import PostureRules

        return replace(PostureRules(), **dict(self.posture_overrides))

    def config_hash(self) -> str:
        return sha256(yaml.safe_dump(dict(self.raw), sort_keys=True).encode()).hexdigest()[:16]


_TOP_LEVEL = {"schema_version", "run_id", "source_root", "model_root", "inventory",
              "metadata_root", "outputs", "clips", "scan", "video", "posture", "legacy"}


def load_config(path: str | Path = DEFAULT_CONFIG) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: top level must be a mapping")
    if int(raw.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"{path}: expected schema_version {SCHEMA_VERSION}")
    unknown = set(raw) - _TOP_LEVEL
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown)}")

    outputs = dict(raw.get("outputs") or {})
    clips_block = dict(raw.get("clips") or {})
    scan_block = dict(raw.get("scan") or {})
    gesture_block = dict(scan_block.pop("gesture", None) or {})
    video_block = dict(raw.get("video") or {})
    posture_block = dict(raw.get("posture") or {})
    legacy_block = dict(raw.get("legacy") or {})

    for name, block, allowed in (
        ("outputs", outputs, {"root"}),
        ("clips", clips_block, {"seconds"}),
        ("scan", scan_block, {"tasks"}),
        ("video", video_block, {"short_side"}),
        ("legacy", legacy_block, {"run_root"}),
    ):
        unknown = set(block) - allowed
        if unknown:
            raise ValueError(f"{path}: unknown {name} keys {sorted(unknown)}")
    unknown = set(gesture_block) - set(GestureParams().__dataclass_fields__)
    if unknown:
        raise ValueError(f"{path}: unknown gesture keys {sorted(unknown)}")
    if posture_block:
        from .posture import PostureRules

        unknown = set(posture_block) - set(PostureRules().__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path}: unknown posture keys {sorted(unknown)}")

    clip_seconds = float(clips_block.get("seconds", 30.0))
    if not 1.0 <= clip_seconds <= 600.0:
        raise ValueError(f"{path}: clips.seconds must be in [1, 600]")
    legacy_root = legacy_block.get("run_root")

    return RunConfig(
        run_id=str(raw["run_id"]),
        source_root=Path(raw["source_root"]),
        model_root=Path(raw.get("model_root", "model_files")),
        inventory=Path(raw["inventory"]),
        metadata_root=Path(raw["metadata_root"]),
        output_root=Path(outputs["root"]),
        clip_seconds=clip_seconds,
        scan_tasks=int(scan_block.get("tasks", 512)),
        gesture=replace(GestureParams(), **gesture_block) if gesture_block else GestureParams(),
        video_short_side=int(video_block.get("short_side", 540)),
        posture_overrides=posture_block,
        legacy_run_root=Path(legacy_root) if legacy_root else None,
        raw=raw,
    )
