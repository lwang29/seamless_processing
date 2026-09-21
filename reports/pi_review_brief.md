# ViBES upper-body subset — 30-clip calibration review

**Ask:** judge 30 clips (about 15 minutes of video, 25 with pauses). Accept = you would
train ViBES on this. The clips are presented blind and in random order; two different
samples are mixed together and the split is revealed after you finish.

**Open the gallery:** see the tunnel command at the end.

---

## What the automated filter measures

Every recording is measured in **30-second windows** (10 s hop). All motion is expressed in
a **torso frame** whose origin is the shoulder midpoint and whose axes come from the
shoulder line and the pelvis-to-neck direction, so swaying, turning and stepping cannot be
counted as gesture. Eighteen named clauses in three groups must all pass:

**1. Upper-body tracking validity** — SMPL-H valid on ≥90% of frames, no invalid run longer
than 1 s, released 2D keypoint confidence p10 ≥ 0.30, <0.2% implausible wrist speeds.
Lower body, seated posture and legs out of frame are not examined at all.

**2. Gesture during speech** — measured only inside the participant's own voice-activity
segments. Requires ≥8 s of own speech; gesture present in ≥35% of speech time; ≥40% of
separate utterances covered; ≥3 distinct gesture episodes (≥0.30 s each, gaps under 0.25 s
merged); wrist excursion ≥80 mm and elbow ≥35 mm in the torso frame; and gesture rate at
least 1.05× higher while speaking than while silent. A gesture frame needs both speed
(≥60 mm/s) and **travel** (≥35 mm within 0.5 s), so jitter in place does not qualify.

**3. Posture variety** — added after the first review round, because activity alone let
through people who gesture without ever moving their arms anywhere: mean pairwise wrist
spread ≥150 mm, hands clasped together <55% of the time, arm abduction p75 ≥17°.

Two independent noise guards compare SMPL-H-derived speed against the released 2D keypoints
(correlation ≥0.45) and check step direction coherence. Neither is a magnitude threshold:
106 of 135 candidate jitter measures turned out to be confounded with genuine gesture
activity (up to rho 0.903), so any magnitude-based jitter gate would preferentially delete
the most active gesturers.

**Scale:** 118,570 files → 2,423,304 windows → **73,883 candidate clips / 616 hours**
(31,815 files, 3,724 participants), all four vendors.

## What the manual stage adds

The automated filter is deliberately not the final answer. Every clip in the accepted
manifest is inspected individually against `docs/review_rubric.md`. Explicitly **not**
reasons to reject: seated posture, legs out of frame, lower-body tracking errors, a hand
briefly leaving the image, or an unusual camera angle.

So far 552 items have a per-item verdict; 100 of those were then re-reviewed by hand with
audio. On those 100:

| | you would accept | you would reject |
|---|---|---|
| **filter + review accepted** | 67 | **0** |
| **filter + review rejected/unsure** | 10 | 23 |

No clip that passed review was judged unusable on re-inspection (n=67; with 95% confidence
the true false-accept rate is under about 4.5%). The review stage is somewhat **too strict**
rather than too permissive — it discards roughly 13% of usable material, nine of ten such
cases labelled "static hands".

Current verified subset: **848 clips / 7.07 hours** across 375 participants. The subset is
limited by review throughput, not by data; the full candidate pool projects to roughly
**407 hours** at the current accept rate.

## The one question we cannot answer without you

**Is one-handed gesturing acceptable?** Several participants gesture actively and in time
with their speech using one hand while the other stays in the lap or on a chair arm. Both
the automated measures and the human reviewer landed on "unsure" for every such case. If
these count, the accept rate and the final hour count both rise appreciably. If they do not,
we add a bimanual clause and lose some data. Please mark these **unsure** rather than
guessing, and say afterwards which way you would like them treated.

Two other patterns we have already decided, for your confirmation:

- **Props.** Some participants hold or flip through a printed prompt sheet. This produces
  either static hands or motion unrelated to speech, and we reject it. It is markedly more
  common in vendor V03 (accept rate 64% vs 85% elsewhere, p=0.027), which is 41% of the pool.
- **Self-adjustment.** Repeatedly adjusting a hat or glasses can produce top-decile motion
  statistics with no communicative gesture at all. We reject it; it is the clearest reason
  the manual stage exists.

## Opening the gallery

```bash
ssh -N -L 8900:localhost:8900 <your-user>@simurgh2.stanford.edu
```

Then open <http://localhost:8900/> and press **V** to play each clip with sound.
**A** accept · **R** reject · **U** unsure · number keys tag a reason · **N** next unjudged.
Verdicts save to the cluster as you go; you can stop and resume at any time.

Participant video stays on the cluster — the server listens on localhost only and the tunnel
is the only way in.
