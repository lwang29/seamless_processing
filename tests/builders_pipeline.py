"""A miniature release on disk, run through the real pipeline, with known answers.

Four interactions, chosen so that every pairing and missing-value rule has a case:

``V00_S0001_I00000129``  P0010 (standing, speaks first) + P0020 (seated, answers);
                         50 s each, so clip 1 is a 20-s partial clip. Only P0010
                         has 3P MOI annotations (coverage 'one').
``V00_S0002_I00000130``  P0030 alone: partner missing (a single-participant session);
                         its WAV is 31 s on a 35-s pose grid, so clip 1's speech is unknown.
``V03_S0001_I00000135``  P0100 whose NPZ/JSON are absent (it sorts first: member 1)
                         + P0101 measured (member 2).
``V01_S0003_I00000139``  P0200 at 30 fps (46 s) + P0201 at 29.97 fps (40 s): the
                         durations disagree by > 1 s, so partners are not linked.
                         P0200's VAD drops out 20-40 s while its words continue
                         (a VAD gap: clip 1's speech is unannotated).

Everything is written as the release writes it (NPZ arrays, JSON with VAD,
transcript and annotations, 48 kHz float WAV), then catalogued with
:func:`catalog.build_catalog`, scanned with :func:`scan.scan_members`, and
annotated with :func:`annotate.build_tables` — the same code paths as a real run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tests.conftest import make_bundle

SAMPLE_RATE = 16_000  # the pipeline reads any rate; 16 kHz keeps the fixture small


@dataclass(frozen=True)
class Spec:
    file_id: str
    label: str
    split: str
    fps: float
    seconds: float
    speech: tuple[tuple[float, float], ...]
    seated: bool = False
    swing_deg: float = 35.0
    moi_3p: tuple[tuple[int, int, str], ...] = ()
    present: bool = True
    wav_seconds: float | None = None  # a WAV shorter than the pose grid
    word_only: tuple[tuple[float, float], ...] = ()  # transcript words with no VAD (a VAD gap)


SPECS: tuple[Spec, ...] = (
    Spec("V00_S0001_I00000129_P0010", "improvised", "train", 30.0, 50.0,
         ((1.0, 8.0), (20.0, 27.0), (40.0, 46.0)),
         moi_3p=((10, 12, "Participant laughs"), (28, 30, "Waves"), (41, 41, "Nods"))),
    Spec("V00_S0001_I00000129_P0020", "improvised", "train", 30.0, 50.0,
         ((9.0, 18.0), (28.0, 38.0)), seated=True, swing_deg=10.0),
    Spec("V00_S0002_I00000130_P0030", "naturalistic", "dev", 30.0, 35.0, ((2.0, 19.0),), wav_seconds=31.0),
    Spec("V03_S0001_I00000135_P0100", "improvised", "train", 30000 / 1001, 45.0, ((1.0, 30.0),), present=False),
    Spec("V03_S0001_I00000135_P0101", "improvised", "train", 30000 / 1001, 45.0, ((31.0, 44.0),)),
    Spec("V01_S0003_I00000139_P0200", "naturalistic", "test", 30.0, 46.0, ((1.0, 10.0), (44.0, 46.0)),
         word_only=((20.0, 40.0),)),
    Spec("V01_S0003_I00000139_P0201", "naturalistic", "test", 30000 / 1001, 40.0, ((12.0, 25.0),)),
)


def _legs(keypoints: np.ndarray) -> None:
    """Hips, knees and ankles inside a 1080x1920 raster, confidently detected."""

    for index, xy in ((11, (600, 1000)), (12, (480, 1000)), (13, (600, 1250)), (14, (480, 1250)),
                      (15, (600, 1500)), (16, (480, 1500)), (0, (540, 560))):
        keypoints[:, index, :2] = xy
        keypoints[:, index, 2] = 0.95


def _payload(spec: Spec, seed: int) -> dict[str, np.ndarray]:
    frames = int(round(spec.seconds * spec.fps))
    bundle = make_bundle(frames=frames, shoulder_swing_deg=spec.swing_deg, speech=spec.speech, seed=seed)
    payload = dict(bundle.payload)
    # The release's root rotation is ~pi about x (camera y points down).
    payload["smplh:global_orient"] = np.tile(np.array([[np.pi, 0.0, 0.0]], np.float32), (frames, 1))
    payload["smplh:translation"] = np.tile(np.array([[0.0, 0.3, 40.0]], np.float32), (frames, 1))
    if spec.seated:
        body = payload["smplh:body_pose"].copy()
        body[:, 0, 0] = body[:, 1, 0] = -np.pi / 2  # hips flexed
        body[:, 3, 0] = body[:, 4, 0] = np.pi / 2   # knees flexed
        payload["smplh:body_pose"] = body
    keypoints = payload["boxes_and_keypoints:keypoints"].copy()
    _legs(keypoints)
    payload["boxes_and_keypoints:keypoints"] = keypoints
    payload["boxes_and_keypoints:box"] = np.tile(np.array([[300.0, 450.0, 800.0, 1600.0]], np.float32), (frames, 1))
    return payload


def _annotation(spec: Spec) -> dict:
    words = []
    for start, end in sorted(spec.speech + spec.word_only):
        t = start
        while t + 0.4 <= end:
            words.append({"word": "gesturing", "start": t, "end": t + 0.35, "score": 0.9})
            t += 0.5
    annotation = {
        "id": spec.file_id,
        "metadata:vad": [{"start": a, "end": b} for a, b in spec.speech],
        "metadata:transcript": [{"start": spec.speech[0][0], "end": spec.speech[-1][1], "transcript": "",
                                 "words": words}],
    }
    if spec.moi_3p:
        for kind in ("3P-IS", "3P-R", "3P-V"):
            annotation[f"annotations:{kind}"] = [
                {"annotation": text, "start_ts": a, "end_ts": b} for a, b, text in spec.moi_3p]
    return annotation


def _wav(spec: Spec, partner: Spec | None, rng: np.random.Generator) -> np.ndarray:
    n = int(round((spec.wav_seconds or spec.seconds) * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    audio = rng.normal(0.0, 1e-4, n)
    for a, b in spec.speech:
        mask = (t >= a) & (t < b)
        audio[mask] += 0.1 * np.sin(2 * np.pi * 180 * t[mask])
    if partner is not None:
        for a, b in partner.speech:  # bleed, 20 dB down
            mask = (t >= a) & (t < b)
            audio[mask] += 0.01 * np.sin(2 * np.pi * 240 * t[mask])
    return audio.astype(np.float32)


def write_release(root: Path) -> tuple[pd.DataFrame, Path]:
    """Write the miniature release and its metadata; return (census, metadata_root)."""

    import soundfile

    rng = np.random.default_rng(0)
    rows = []
    by_interaction: dict[str, list[Spec]] = {}
    for spec in SPECS:
        by_interaction.setdefault(spec.file_id.rsplit("_P", 1)[0], []).append(spec)
    for index, spec in enumerate(SPECS):
        relbase = f"{spec.label}/{spec.split}/0000/{index:04d}/{spec.file_id}"
        base = root / "release" / relbase
        base.parent.mkdir(parents=True, exist_ok=True)
        mates = [s for s in by_interaction[spec.file_id.rsplit("_P", 1)[0]] if s is not spec]
        if spec.present:
            np.savez(base.with_suffix(".npz"), **_payload(spec, seed=index))
            base.with_suffix(".json").write_text(json.dumps(_annotation(spec)), encoding="utf-8")
            soundfile.write(str(base.with_suffix(".wav")), _wav(spec, mates[0] if mates else None, rng),
                            SAMPLE_RATE, subtype="FLOAT")
        frames = int(round(spec.seconds * spec.fps))
        rows.append({
            "file_id": spec.file_id, "label": spec.label, "split": spec.split, "batch_idx": 0,
            "archive_idx": index, "has_imitator_movement": 0, "has_annotation_1p": 0,
            "has_annotation_3p": int(bool(spec.moi_3p)), "source_relbase": relbase,
            "json_present": spec.present, "mp4_present": spec.present, "npz_present": spec.present,
            "wav_present": spec.present, "probe_status": "ok", "video_width": 1080.0, "video_height": 1920.0,
            "video_r_fps": spec.fps, "video_nb_frames": float(frames), "video_duration_s": frames / spec.fps,
            "mp4_size_bytes": 10_000_000.0,
        })
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        ("00000129", "P0002_v3.4.aac_ANCP_XXXX-148", "Read each sentence aloud and act-out a corresponding gesture.",
         "Read each sentence aloud and act-out a corresponding gesture.", "ANCP", "", "grounded_gesture"),
        ("00000130", "P0070_v3.4.fam_ANCP_XXXX", "Tell your partner about a time you felt proud.",
         "Your partner will tell you about a time they felt proud.", "ANCP", "", "ipc_conversation"),
        ("00000135", "RP2.0_AMCP_ANCP", "Scenario: You are a landlord. Negotiate.", "Scenario: You are a tenant.",
         "AMCP", "ANCP", "ipc_conversation"),
        ("00000139", "P0100_v3.1.str_APCN_XXXX", "Discuss a topic you feel strongly about.",
         "Your partner will discuss a topic.", "APCN", "", "ipc_conversation"),
    ], columns=["prompt_hash", "prompt_id_unique", "participant_a_prompt_text", "participant_b_prompt_text",
                "ipc_a", "ipc_b", "interaction_type"]).to_csv(meta / "interactions.csv", index=True)
    bfi = ["extraversion_raw", "agreeableness_raw", "conscientiousness_raw", "neuroticism_raw", "openness_raw"]
    people = [("00", "0010", "3.5"), ("00", "0020", "Undisclosed"), ("00", "0030", "4.0"),
              ("03", "0100", "2.5"), ("03", "0101", "3.0"), ("01", "0200", "4.5")]  # 0201: no row
    pd.DataFrame([{**{b: value for b in bfi}, "vendor_id": v, "participant_id": p} for v, p, value in people]).to_csv(
        meta / "participants.csv", index=False)
    pd.DataFrame([("V00", "0001", "familiar", "friends"), ("V01", "0003", "stranger", "stranger")],
                 columns=["vendor_id", "session_id", "relationship", "relationship_detail"]).to_csv(
        meta / "relationships.csv", index=False)
    return pd.DataFrame(rows), meta


def run_pipeline(root: Path, model_root: Path, clip_seconds: float = 30.0):
    """Catalog -> scan -> annotate -> validate. Returns (tables, report, catalog)."""

    from seamless_curation import annotate, catalog, posture, scan, validate

    census, meta = write_release(root)
    cat = catalog.build_catalog(census, meta)
    settings = scan.ScanSettings(clip_seconds=clip_seconds, model_root=str(model_root))
    frames = scan.scan_members(cat.recordings, root / "release", settings)
    scanned = annotate.ScanTables(**frames)
    tables, _reference = annotate.build_tables(cat.tables(), scanned, rules=posture.PostureRules(),
                                               clip_seconds=clip_seconds)
    report = validate.validate_tables(tables, clip_seconds=clip_seconds,
                                      catalog_file_ids=pd.Index(cat.recordings["file_id"]))
    return tables, report, cat
