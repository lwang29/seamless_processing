"""``seamless-curation stats`` — the run's numbers, regenerated from its outputs.

Every table in ``reports/17_cospeech_gesture.md`` comes from here rather than
from a transcription, so the report cannot drift from the artefacts it describes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RunConfig
from .corpus import eligibility_counts
from .review_store import VerdictStore


def _table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in record) + " |"
        for record in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *rows])


def write_report(config: RunConfig, out: Path | None = None) -> Path:
    if not config.population_path.exists():
        raise SystemExit(
            f"{config.population_path} does not exist: run `seamless-curation population` first. "
            "Every later section of the report is optional and is skipped if its stage has not run."
        )
    out = out or (config.output_root / "run_report.md")
    lines: list[str] = [f"# {config.run_id} — run report", ""]
    lines += [
        f"Config hash `{config.config_hash()}`, scan fingerprint "
        f"`{config.scan.fingerprint()}`. Regenerate with `seamless-curation stats`.",
        "",
        "The scan fingerprint identifies the measurement parameters **and** the source of "
        "`gesture.py` and `smplh_kinematics.py`; every shard marker under `scan_shards/` "
        "carries it, so the numbers below can be traced to the code that produced them.",
        "",
    ]

    population = pd.read_parquet(config.population_path)
    counts = eligibility_counts(population)
    lines += ["## 1. Population", "", f"{counts.total:,} participant files, {counts.eligible:,} eligible.", ""]
    reasons = pd.DataFrame(
        sorted(counts.by_reason.items(), key=lambda kv: -kv[1]), columns=["ineligible_reason", "files"]
    )
    if len(reasons):
        lines += [_table(reasons), ""]
    by_vendor = (
        population.groupby("vendor")
        .agg(files=("file_id", "size"), eligible=("eligible", "sum"),
             hours=("observed_duration_s", lambda s: round(s.sum() / 3600, 1)))
        .reset_index()
    )
    lines += [_table(by_vendor), ""]

    if config.windows_path.exists():
        windows = pd.read_parquet(config.windows_path)
        lines += [
            "## 2. Scan",
            "",
            f"{len(windows):,} windows of {config.scan.window_seconds:.0f} s "
            f"at a {config.scan.hop_seconds:.0f} s hop.",
            "",
        ]
    funnel_path = config.output_root / "gate_funnel.csv"
    if funnel_path.exists():
        funnel = pd.read_csv(funnel_path)
        funnel["share"] = (funnel["share"] * 100).round(1).astype(str) + "%"
        lines += ["## 3. Gate funnel", "", "First failing clause per window.", "", _table(funnel), ""]

    if config.candidates_path.exists():
        candidates = pd.read_parquet(config.candidates_path)
        items = pd.read_csv(config.review_manifest_path)
        lines += [
            "## 4. Candidates",
            "",
            f"{len(candidates):,} candidate clips over {items.shape[0]:,} review items "
            f"({candidates['window_seconds'].sum() / 3600:,.0f} h), "
            f"{items['participant_key'].nunique():,} participants.",
            "",
            _table(
                items.groupby("vendor")
                .agg(review_items=("review_item_id", "size"),
                     hours=("clip_seconds", lambda s: round(s.sum() / 3600, 1)))
                .reset_index()
            ),
            "",
        ]

    resolved = VerdictStore(config.verdict_log).resolve()
    lines += ["## 5. Manual review", ""]
    if resolved.empty:
        lines += ["No verdicts recorded yet.", ""]
    else:
        verdicts = resolved["verdict"].value_counts().rename_axis("verdict").reset_index(name="items")
        lines += [f"{len(resolved):,} review items judged.", "", _table(verdicts), ""]
        rejects: dict[str, int] = {}
        for reasons_list in resolved.loc[resolved["verdict"] == "reject", "reasons"]:
            for reason in reasons_list or ["unspecified"]:
                rejects[reason] = rejects.get(reason, 0) + 1
        if rejects:
            lines += [
                "Reject reasons:",
                "",
                _table(pd.DataFrame(sorted(rejects.items(), key=lambda kv: -kv[1]),
                                    columns=["reason", "items"])),
                "",
            ]
        sources = resolved["verdict_source"].value_counts().rename_axis("verdict_source").reset_index(name="items")
        lines += ["Verdict sources:", "", _table(sources), ""]

    if config.accepted_clips_path.exists():
        accepted = pd.read_csv(config.accepted_clips_path)
        lines += ["## 6. Accepted subset", ""]
        if accepted.empty:
            lines += ["Empty: nothing has been accepted yet.", ""]
        else:
            lines += [
                f"{len(accepted):,} clips, {accepted['window_seconds'].sum() / 3600:,.1f} h, "
                f"{accepted['file_id'].nunique():,} files, "
                f"{(accepted['vendor'].astype(str) + ':' + accepted['participant_id'].astype(str)).nunique():,} "
                "participants.",
                "",
                _table(
                    accepted.groupby(["vendor", "label"])
                    .agg(clips=("clip_id", "size"),
                         hours=("window_seconds", lambda s: round(s.sum() / 3600, 2)))
                    .reset_index()
                ),
                "",
            ]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
