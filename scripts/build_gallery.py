"""Build a self-contained static gallery of randomly sampled accepted clips.

The point is quality assessment by someone who will not run this pipeline: a
large uniform random sample of the accepted set, each clip playable with its own
audio, with enough metadata visible to tell *why* the filter kept it.

The sample is **uniform random with a fixed seed**, not stratified and not
curated. That is the whole methodological point — a hand-picked or
quality-sorted sample would flatter the output and answer a question nobody
asked. The page prints the sample's composition against the full accepted set so
a reader can confirm it is representative rather than take it on trust.

Usage::

    PYTHONPATH=src python scripts/build_gallery.py --out /path/to/build --count 400
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seamless_curation.dataset import load_manifest  # noqa: E402
from seamless_curation.review_card import upper_body_crop_box  # noqa: E402

VIDEO_W, VIDEO_H = 360, 480
POSTER_W, POSTER_H = 180, 240


def _probe_raster(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    width, height = (int(value) for value in out.split("x")[:2])
    return width, height


def render_one(job: dict) -> dict:
    """Crop to the upper body, cut the frame range, encode for the web."""

    source = Path(job["source_root"]) / job["source_relbase"]
    video, audio, npz = source.with_suffix(".mp4"), source.with_suffix(".wav"), source.with_suffix(".npz")
    out_dir = Path(job["out"])
    clip_path = out_dir / "clips" / f"{job['slug']}.mp4"
    poster_path = out_dir / "posters" / f"{job['slug']}.jpg"
    if clip_path.exists() and poster_path.exists() and not job["overwrite"]:
        return {"slug": job["slug"], "status": "cached"}

    try:
        with np.load(npz) as archive:
            keypoints = archive["boxes_and_keypoints:keypoints"][job["start_frame"]:job["end_frame"]]
        width, height = _probe_raster(video)
        # One box for the whole clip: a crop that chases the hands turns a
        # gesture into a zoom and makes the motion impossible to judge.
        sampled = list(range(0, len(keypoints), max(1, len(keypoints) // 30)))
        x, y, w, h = upper_body_crop_box(keypoints, sampled, width, height, pad=0.18, aspect=0.75)
    except Exception as error:  # noqa: BLE001 - one bad clip must not kill the build
        return {"slug": job["slug"], "status": f"prepare-failed: {type(error).__name__}: {error}"}

    chain = f"crop={w}:{h}:{x}:{y},scale={VIDEO_W}:{VIDEO_H}:flags=bicubic"
    common = ["-ss", str(job["start_s"]), "-t", str(job["seconds"])]
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             *common, "-i", str(video), *common, "-i", str(audio),
             "-filter:v", chain, "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-movflags", "+faststart",
             str(clip_path)],
            check=True, capture_output=True,
        )
        # Poster from a third of the way in, so it is rarely a rest pose.
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-ss", str(job["start_s"] + job["seconds"] / 3), "-i", str(video),
             "-frames:v", "1", "-filter:v",
             f"crop={w}:{h}:{x}:{y},scale={POSTER_W}:{POSTER_H}:flags=bicubic",
             "-q:v", "6", str(poster_path)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        return {"slug": job["slug"], "status": f"ffmpeg: {error.stderr.decode()[:160]}"}
    return {"slug": job["slug"], "status": "ok", "bytes": clip_path.stat().st_size}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/vibes_upper_body_v1/export/clips_accepted.csv")
    parser.add_argument("--source-root", default="seamless_interaction")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", default="gallery-v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "posters").mkdir(parents=True, exist_ok=True)

    everything = load_manifest(args.manifest)
    # Uniform random over clips, seeded. Not stratified: the question is what
    # the accepted set looks like, and any weighting would answer a different one.
    seed = int.from_bytes(args.seed.encode(), "big") % (2**32)
    sample = everything.sample(min(args.count, len(everything)), random_state=seed)
    sample = sample.sort_values("gesture_quality", ascending=False).reset_index(drop=True)
    sample["slug"] = [f"c{i:04d}" for i in range(len(sample))]

    jobs = [
        {
            "slug": row.slug, "source_root": args.source_root, "source_relbase": row.source_relbase,
            "start_frame": int(row.start_frame), "end_frame": int(row.end_frame),
            "start_s": float(row.start_s), "seconds": float(row.window_seconds),
            "out": str(out), "overwrite": args.overwrite,
        }
        for row in sample.itertuples()
    ]
    print(f"rendering {len(jobs)} clips with {args.workers} workers ...", flush=True)
    with mp.Pool(args.workers) as pool:
        results = []
        for index, result in enumerate(pool.imap_unordered(render_one, jobs), 1):
            results.append(result)
            if index % 25 == 0 or index == len(jobs):
                ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
                print(f"  {index}/{len(jobs)}  ok={ok}", flush=True)

    status = {r["slug"]: r["status"] for r in results}
    failed = {s: v for s, v in status.items() if v not in ("ok", "cached")}
    if failed:
        print(f"\n{len(failed)} clips failed to render; they are dropped from the gallery:")
        for slug, why in list(failed.items())[:5]:
            print(f"  {slug}: {why}")
    sample = sample.loc[sample.slug.map(lambda s: status.get(s) in ("ok", "cached"))].reset_index(drop=True)

    payload = write_site(out, sample, everything, args)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\nwrote {len(sample)} clips to {out}  ({total / 1e6:.0f} MB)")
    return 0 if not failed else 0


def write_site(out: Path, sample: pd.DataFrame, everything: pd.DataFrame, args) -> dict:
    from gallery_page import render_page  # noqa: PLC0415

    items = [
        {
            "slug": row.slug,
            "file_id": row.file_id,
            "vendor": row.vendor,
            "split": row.split,
            "seconds": round(float(row.window_seconds), 1),
            "quality": round(float(row.gesture_quality), 3),
            "posture": round(float(row.dim_posture), 2),
            "persistence": round(float(row.dim_persistence), 2),
            "vigour": round(float(row.dim_vigour), 2),
            "integrity": round(float(row.dim_integrity), 2),
            "speech_s": round(float(row.speech_seconds), 1),
            "gesture_frac": round(float(row.gesture_frac_speech), 2),
            "start_s": round(float(row.start_s), 1),
        }
        for row in sample.itertuples()
    ]

    def dist(frame: pd.DataFrame) -> dict:
        return {
            "clips": int(len(frame)),
            "vendors": {k: round(v, 3) for k, v in frame.vendor.value_counts(normalize=True).items()},
            "quality": [round(float(frame.gesture_quality.quantile(q)), 3) for q in (0.1, 0.5, 0.9)],
            "hours": round(float(frame.window_seconds.sum() / 3600), 1),
        }

    context = {
        "items": items,
        "sample": dist(sample),
        "population": dist(everything),
        "seed": args.seed,
        "count_requested": args.count,
    }
    (out / "items.json").write_text(json.dumps(context, indent=1), encoding="utf-8")
    (out / "index.html").write_text(render_page(context), encoding="utf-8")
    return context


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
