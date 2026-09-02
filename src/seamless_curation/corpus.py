"""The eligible population: which participant files the gesture scan may read.

Eligibility here is only about whether a file *can* be measured and whether its
source is usable at all for upper-body training. It says nothing about gesture;
that is :mod:`seamless_curation.gates`.

Four exclusions, each with the reason it exists:

``incomplete``
    The bundle is missing one of ``.json/.mp4/.npz/.wav``, or ``ffprobe`` found
    no video stream. 439 files in the corpus have no video stream and the
    261-byte-MP4 size signature misses 154 of them, so this is probe-based.

``excluded_raster``
    Rasters the reviewer ruled out in Round 8 and that remain ruled out for
    upper-body training. V01's anamorphic 2160x2160 and 1920x1080 storage is the
    important case: the picture and the released 2D points can be un-squeezed,
    but the released SMPL-H **cannot**, because the fitted camera is isotropic
    and the pose absorbed the stretch — so the very parameters ViBES would train
    on are wrong. V03's 640x480 and 3840x2160 are room cameras: far-field audio,
    which makes co-speech alignment meaningless.

``timebase_drift``
    The container's frame count and its nominal rate disagree with the *video
    stream's* own duration by more than half a second. Two such files were
    checked by hand and the SMPL-H tracking was wholly unrelated to the video in
    both. Note what this is not: comparing against ``format_duration_s`` instead
    compares the container with its own **audio** track, which is routinely a
    few hundred milliseconds longer than the video and has nothing to do with
    the annotation grid — it marked three files ineligible whose npz frame count
    matched their video exactly.

``no_speech_activity``
    ``charades`` interactions are gestural but speech-free by design, so they
    cannot contribute *co-speech* gesture. They are excluded from the candidate
    population rather than silently failing the speech gate later.

Everything the previous pipeline rejected for **posture** is deliberately absent.
Seated participants, legs out of frame, and hands that leave the raster are all
eligible: ViBES trains the upper body only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Rasters whose released SMPL-H or audio cannot support upper-body training.
EXCLUDED_RASTERS: tuple[str, ...] = ("2160x2160", "1920x1080", "640x480", "3840x2160")

#: Interaction types with no conversational speech.
EXCLUDED_INTERACTION_TYPES: tuple[str, ...] = ("charades",)

#: Released grids further than this from the container are not merely a
#: rendering nuisance.
MAX_TIMEBASE_DRIFT_S = 0.5

#: A file shorter than this cannot hold even one candidate window.
MIN_DURATION_S = 40.0

POPULATION_COLUMNS = [
    "file_id", "vendor", "label", "split", "source_relbase",
    "session_id", "participant_id", "interaction_id", "interaction_type",
    "raster", "nominal_fps", "observed_duration_s", "npz_size_bytes",
    "eligible", "ineligible_reason",
]


@dataclass(frozen=True)
class EligibilityCounts:
    total: int
    eligible: int
    by_reason: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {"total": self.total, "eligible": self.eligible, "by_reason": self.by_reason}


def build_population(inventory: pd.DataFrame) -> pd.DataFrame:
    """Annotate the M-1 inventory with eligibility, one row per participant file."""

    frame = inventory.copy()
    frame["raster"] = (
        frame["video_width"].astype("Int64").astype(str)
        + "x"
        + frame["video_height"].astype("Int64").astype(str)
    )
    frame["nominal_fps"] = pd.to_numeric(frame["video_r_fps"], errors="coerce")
    duration = pd.to_numeric(frame["observed_duration_s"], errors="coerce")
    frame_count = pd.to_numeric(frame["video_nb_frames"], errors="coerce")
    annotation_s = frame_count / frame["nominal_fps"]
    video_s = pd.to_numeric(
        frame.get("video_duration_s", pd.Series(index=frame.index, dtype=float)), errors="coerce"
    ).fillna(duration)
    # A file whose probe reported no frame count cannot be checked; it fails
    # rather than bypassing the clause.
    drift = (annotation_s - video_s).abs().where(frame_count.notna(), np.inf)

    reason = pd.Series("", index=frame.index, dtype=object)

    def mark(mask: pd.Series, label: str) -> None:
        reason.loc[mask & (reason == "")] = label

    mark(~frame["all_modalities_present"].astype("boolean").fillna(False).astype(bool), "incomplete_bundle")
    mark(~frame["video_stream_present"].astype("boolean").fillna(False).astype(bool), "no_video_stream")
    mark(frame["probe_status"].fillna("") != "ok", "probe_failed")
    mark(frame["raster"].isin(EXCLUDED_RASTERS), "excluded_raster")
    mark(~frame["nominal_fps"].between(20, 61), "unusable_frame_rate")
    mark(drift > MAX_TIMEBASE_DRIFT_S, "timebase_drift")
    mark(frame["interaction_type"].fillna("").isin(EXCLUDED_INTERACTION_TYPES), "no_speech_activity")
    mark(duration.fillna(0) < MIN_DURATION_S, "too_short")

    frame["eligible"] = reason == ""
    frame["ineligible_reason"] = reason.where(reason != "", None)
    return frame.reindex(columns=POPULATION_COLUMNS)


def eligibility_counts(population: pd.DataFrame) -> EligibilityCounts:
    reasons = population.loc[~population["eligible"], "ineligible_reason"].value_counts()
    return EligibilityCounts(
        total=int(len(population)),
        eligible=int(population["eligible"].sum()),
        by_reason={str(k): int(v) for k, v in reasons.items()},
    )


def shard_of(file_ids: pd.Series, shards: int) -> pd.Series:
    """Deterministic shard assignment by file id, independent of row order.

    Restart safety depends on a task always landing in the same shard, so this
    must not use the DataFrame index.
    """

    if shards < 1:
        raise ValueError("shards must be >= 1")
    digest = file_ids.astype(str).map(lambda s: int.from_bytes(_blake(s), "big"))
    return (digest % shards).astype("int32")


def _blake(text: str) -> bytes:
    from hashlib import blake2b

    return blake2b(text.encode("utf-8"), digest_size=8).digest()


def stable_key(*parts: str, length: int = 14) -> str:
    """A short, stable, order-independent identifier for a tuple of strings.

    Used for ``review_item_id``. It has to be a function of the *content* it
    names: an identifier derived from a row's position re-binds every stored
    verdict when the rows change.
    """

    from hashlib import blake2b

    digest = blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8).hexdigest()
    return digest[:length]


def stable_unit_interval(*parts: str, salt: str = "") -> float:
    """A reproducible pseudo-random number in [0, 1) from string parts.

    Used for seeded sampling that must not depend on row order, numpy version or
    platform.
    """

    from hashlib import blake2b

    digest = blake2b("\x1f".join((salt,) + parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)
