# Round 7: what your V01/V02/V03 observations turned out to be

Every item you raised, what it measured out as, and what I did about it. The
short version:

| you saw | what it is | fixed? |
|---|---|---|
| 31 V01 clips horizontally stretched | **anamorphic storage.** The container declares a 9:16 display aspect for *every* V01 raster; only 1080×1920 stores square pixels. | **Picture and 2D points: yes.** Released SMPL-H: **no, and it cannot be** — §1.4. |
| those same clips tripping FM1 | consequence of the above. The fit bent the body to match a stretched person. | follows from the above |
| V03 clips rotated a quarter turn | **baked into the pixels**, no rotation tag. 1,396 eligible files. | **Yes, exactly.** A rigid transform — §2. |
| V03 sitters FM1 missed | **calibration, not breakage.** Missed 0.435–0.586, caught 0.301–0.429, cut at 0.43. | needs your labels — **ASK 1** |
| FM1 firing on standing people with legs cut off | **the shin clause, not the knee clause.** All eleven. | measured and surfaced; verdict unchanged — §4 |
| keypoints desynced from the video | **the annotation grid and the container disagree**, so pairing by frame number drifts. | **Yes** — median error 4.45 s → 0 ms, §5 |
| progress bar snapping back | **three keyframes in a 30 s clip**, and `http.server` has no range support. | **Yes**, both halves — §7 |

Two things I found while chasing these that you did not ask about: 183 eligible
files ship an **empty WAV** (§6), and that was silently truncating rendered clips.

New gallery: `artifacts/private_review_vendors_r7/`. Briefing layout throughout,
which is now the standard. Five **ASK** sections at the top carry what I need
from you; the rest is a fresh random draw stratified by vendor and by exactly
which checks fired.

---

## 1. The V01 stretch is in the container, and it says so

### 1.1 What the metadata says

Your hypothesis — "these are all the clips that are in 2160×2160, and could
yield valid video when un-stretched back to 1080×1920" — is exactly right, and
the container agrees in writing. All 31 clips you listed are 2160×2160
(`fm1_sitting` fires on 30 of them, 96.8%).

Probing the pixel aspect ratio across every raster in the corpus:

| vendor | raster | eligible files | pixel aspect | display aspect | needs correcting? |
|---|---|---|---|---|---|
| V00 | 1080×1920 | 41,205 | 1:1 | 9:16 | no |
| **V01** | 1080×1920 | 6,432 | 1:1 | 9:16 | no |
| **V01** | **2160×2160** | **5,002** | **9:16** | 9:16 | **stored 1.78× too wide** |
| **V01** | **1012×1920** | **102** | **270:253** | 9:16 | stored 0.94× — 6.7% too narrow |
| **V01** | **1920×1080** | **60** | **81:256** | 9:16 | **stored 3.16× too wide** |
| V02 | 1080×1920 | 29,502 | 1:1 | 9:16 | no |
| V03 | 2160×3840 | 41,409 | 1:1 | 9:16 | no |
| V03 | 3840×2160 | 1,336 | 1:1 | 16:9 | **quarter turn** (§2) |
| V03 | 640×480 | 60 | 1:1 | 4:3 | **quarter turn** (§2) |
| V03 | 1080×960 | 28 | 1:1 | 9:8 | no — upright, just wide |
| V03 | 2180×3840 | 20 | 1:1 | 109:192 | no |
| V03 | 1080×1920 | 28 | 1:1 | 9:16 | no |

So **every V01 file is portrait content**, and three of its four rasters are
stored horizontally stretched. The pixel aspect is constant within a raster (60
files sampled per raster, one distinct value each).

**This answers your question about 1012×1920 and 1920×1080 directly: both are
anamorphic, both correct to 9:16, and neither needs rotating.** 1012×1920 ×
270/253 = exactly 1080. 1920×1080 × 81/256 = 607.5, so the content is a 607×1080
portrait frame stored across 1920 pixels — it is not landscape footage at all,
which is why it read `knee_between_torso` = −0.303 in Round 6.

### 1.2 The keypoints are in stored coordinates, and the correction fixes them

Shoulder width over torso length in the released 2D keypoints, per raster. Both
distances lie on the same rigid torso, so their ratio is fixed by anatomy up to
how the person is turned, and a horizontal stretch inflates the numerator only.
60 files per raster:

| raster | stored ratio | ÷ V00 | predicted from pixel aspect | after correction |
|---|---|---|---|---|
| V00 1080×1920 | 0.721 | 1.000 | 1.000 | 0.721 |
| V01 1080×1920 | 0.667 | 0.925 | 1.000 | 0.667 |
| **V01 2160×2160** | **1.103** | **1.530** | 1.779 | **0.624** |
| **V01 1920×1080** | **1.786** | **2.477** | 3.165 | **0.573** |
| V01 1012×1920 | 0.683 | 0.947 | 0.937 | 0.728 |
| V02 1080×1920 | 0.628 | 0.871 | 1.000 | 0.628 |
| V03 2160×3840 | 0.642 | 0.890 | 1.000 | 0.642 |

Every square-pixel raster sits in a 0.589–0.721 band. The two badly anamorphic
ones sit far outside it and **land back inside once the pixel aspect is applied**
(0.624 and 0.573). The predicted and observed inflation do not match exactly —
1.53 against 1.78 — because participants are not all square-on to the camera, but
the direction and the correction are unambiguous.

### 1.3 So the video and the keypoints are repairable

`src/seamless_curation/media_repair.py` applies the pixel aspect and any quarter
turn to decoded frames, to released 2D keypoints, to boxes, and to camera-frame
3D joints, with the ffmpeg filter equivalent for the filmstrip. The renderer now
runs everything through it, so **the clips in the new gallery are shown
corrected**, and each card says what was applied. The point transforms are tested
against what OpenCV does to the pixels rather than against a hand-derived
formula, because a keypoint one pixel off a rotated frame would read to a
reviewer as a tracking failure.

### 1.4 But the released SMPL-H is not repairable, and that is the important part

You asked that the correction "occur first in the pipeline, and then the
resulting video should still go through the FM flag processing". That works for
the video and the 2D keypoints. It does not work for the SMPL-H, which is the
training target.

An anisotropic squeeze is not a rigid transform, and the fitted camera is
isotropic — one focal length for both axes. A single 3D body therefore *cannot*
project to a horizontally stretched 2D person. The only way the optimiser could
match one was to distort the pose itself.

Measured. Align the projected body to the released keypoints by translation and
one isotropic scale — shape only, so neither camera centring nor overall scale
can flatter either side — and take the median residual:

| raster | vs keypoints as stored | vs keypoints corrected |
|---|---|---|
| V00 1080×1920 | 0.059 | 0.059 |
| V01 1080×1920 | 0.054 | 0.054 |
| V03 2160×3840 | 0.074 | 0.074 |
| **V01 2160×2160** | **0.113** | **0.218** |
| **V01 1920×1080** | **0.202** | **0.473** |

The released pose matches the *stretched* keypoints about twice as well as the
corrected ones. It is bound to the stretched image. Un-stretching the video is
correct and worth doing; it does not recover the annotation.

Note also that even against the stored keypoints the anamorphic rasters fit
roughly twice as badly as any square-pixel one — the optimiser was straining at
an impossible body and only partly succeeding.

**Consequence.** V01's 5,002 anamorphic 2160×2160 files — **139.0 of its 328.8
eligible dyad-hours** — have unusable released SMPL-H. The options are to drop them,
or to keep them pending a re-fit with a correctly configured camera. The 60
1920×1080 files (1.5 h) are worse and much rarer. The 102 1012×1920 files (1.9 h) are fine: a
6.7% correction, and the shape residual barely moves (0.055 → 0.063, both inside
the normal band). **ASK 2 and ASK 4** in the gallery put this to you.

---

## 2. The V03 rotation, and the 640×480 question

Measuring the in-image angle of the shoulder-to-hip axis over every non-portrait
eligible file:

| raster | files | median roll | one way | the other | upright |
|---|---|---|---|---|---|
| 3840×2160 | 1,336 | −90.00° | 1,215 | 112 | 9 |
| 640×480 | 60 | −92.58° | 60 | 0 | 0 |
| 1080×960 | 28 | +0.99° | 0 | 0 | 28 |

Your split is confirmed: both directions occur in 3840×2160, and the "ground on
the left" case is the common one (1,215 of 1,336). **640×480 is the same defect
at low resolution** — all 60 turned the same way, a quarter-turned room camera.
**1080×960 is not rotated at all**: 28 files, upright, just an unusually wide
frame with a small participant.

Unlike the stretch, **this one is completely repairable.** A quarter turn about
the principal point of a centred pinhole camera with square pixels is exactly a
rotation of the camera about its optical axis, so it maps cleanly onto the
released data: keypoints and boxes rotate, `global_orient` absorbs the turn,
`body_pose` and the hand poses are untouched. The rotated rasters also carry
*normal* anatomy (shoulder/torso 0.653 and 0.628) and *normal* fit quality (shape
residual 0.072 and 0.111), so what is underneath is healthy data.

**It changes no flag.** FM1 is rotation-invariant by construction, FM3 is
normalised and self-anchored, and FM2 reads a released field. That was true
before the repair and is true after it. What the repair buys is that the clips
can be reviewed, the video is usable, and anything downstream that cares about
image orientation now gets the right answer. Round 6 reported 1,387 affected
files from a 7,051-file measurement; the exact eligible count is **1,396 files,
43.8 of V03's 1,429.6 dyad-hours**.

---

## 3. V03's missed sitters: the cut is in the wrong place

Your fourteen labels separate almost perfectly on the measure FM1 cuts on:

| | `knee_between_torso` range | shin direction | FM1 |
|---|---|---|---|
| **caught** (7 files) | 0.301 – **0.429** | all positive | fires |
| **missed** (7 files) | **0.435** – 0.586 | all positive | silent |

The cut is 0.43. Every missed sitter sits just above it and every caught one just
below. The shins are the right way up in all fourteen, so §4's mechanism is not
involved — this is a straightforward posture question with the boundary in the
wrong place.

So FM1 is not broken on V03; it is mis-set, exactly as you suspected. What it
needs is labels on **both** sides of wherever the new cut might go — labelling
only sitters would move it up until it swallowed the standing population. **ASK 1**
draws 6 clips from each of seven bands spanning 0.20 to 0.80, excluding files
whose shins point the wrong way, so each answer lands where the decision is. With
those labels the cut can be re-fitted for V03 the way 0.43 was fitted for V00.

I have deliberately **not** guessed at a new value in the meantime.

---

## 4. FM1 on standing people whose legs are cut off

All eleven clips you listed — 8 in V02, 3 in V03 — fire for the same reason, and
it is not the reason FM1 reports. Their `knee_between_torso` runs 0.552 to 0.677,
comfortably *above* the 0.43 cut, so the knee clause never fires. They fire on
the add-on clause, `shin_inverted`: the shin appears to point upward. And all
eleven have `fm2_feet_out_of_frame_frame_frac` = 1.0 — the feet are never in
shot. With the ankles outside the frame the fit puts them somewhere arbitrary and
the shin comes out inverted.

Across the Round-6 sample of 3,594 files the two clauses split cleanly:

| reason | files |
|---|---|
| `knee_between` | 600 |
| `shin_inverted` | 126 |
| both | 8 |

You said this is "likely fine but something to keep in mind", so **I have not
changed the verdict.** What I have done is make it legible: a new measurement,
`fm1_ankles_visible_frac`, taken from the released 2D keypoints rather than from
the fit, so it is independent of whatever the fit did with the missing legs. It
now appears on every card, next to the shin direction. When you see FM1 fire with
a negative shin and a low ankle-visibility figure, that is this case — FM1 never
saw the legs, rather than having seen them and judged them.

If you later want the verdict to change, the gate is one line: skip the
`shin_inverted` clause when the ankles are out of frame. It would move files that
currently fail FM1 into whatever FM2 and FM3 say about them.

---

## 5. The desynced clip

`V01_S1607_I00000135_P2569`, as released:

| | |
|---|---|
| nominal frame rate (`r_frame_rate`) | 48000/1001 = 47.952 |
| average frame rate | 41.276 |
| frames stored in the container | **2,807** |
| frames in the released annotation arrays | **3,261** |
| duration | 68.005 s |

Both grids cover the same 68.005 s. The annotation grid is uniform at the nominal
rate; the container's is not. So the released arrays and the video frames are two
different samplings of the same interval, and **anything that pairs them by
integer index walks off progressively** — which is exactly what you saw.

Quantified on that file, over the 1,439 frames of a 30 s clip:

| lookup | median error | worst |
|---|---|---|
| by frame number *(what the renderer did)* | **4.45 s** | 6.87 s |
| by timestamp *(what it does now)* | **0 ms** | 41.7 ms — one frame period |

The renderer now works entirely in the annotation timebase: `start_frame` indexes
the released arrays, the clip is encoded at the nominal rate, and each output
frame pulls the video frame that was on screen at that annotation's timestamp,
holding a frame across a gap rather than letting the grids slide apart. For every
file where the two grids agree — which is nearly all of them — this is the same
sequential read as before.

**Prevalence.** 70 of 125,184 eligible files have the container and the nominal
grid disagreeing by more than 0.5 s. Reading the annotation array lengths
directly splits them into four causes:

| cause | files | median drift |
|---|---|---|
| empty annotations (0 frames) — the already-known V01 defect | 35 | — |
| **video short of the nominal grid (dropped frames)** — the case above | **16** | 8.9 s |
| annotation array several times longer than the video | 12 | 10.0 s |
| smaller mismatches | 7 | 1.5 s |

All in V01 except 3 in V02; none in V00 or V03. Of 160 control files, 98.8% have
annotation and container counts within one frame of each other. The scan now
records `timebase_index_drift_s` and `timebase_consistent` for every file, and
**ASK 5** shows the affected clips rendered with the fix so you can confirm the
keypoints now stay put.

I have **not** treated this as a reason to reject a file. The released arrays are
internally consistent; only the pairing with video was wrong, and that is fixed.

---

## 6. Something you did not ask about: 183 files ship no audio

Found because the fix in §5 produced a 20-second clip where it should have
produced 30. `V01_S1607_I00000135_P2569.wav` is **58 bytes** — a RIFF header with
no samples — and the mux used `-shortest`, so the empty audio track silently
truncated the video.

| vendor | eligible files with a ≤1 KiB WAV | with a WAV >1 s shorter than the video |
|---|---|---|
| V00 | **0** | **0** |
| V01 | 61 | 67 |
| V02 | 64 | 90 |
| V03 | 58 | 69 |

0.15% of the eligible pool, and zero in V00, which is why it never came up. The
renderer now detects an empty WAV, substitutes generated silence, and pins the
output length with `-t` and `apad` instead of `-shortest`, so a short audio track
can no longer decide how much video a reviewer sees. The scan records
`audio_empty`, and it shows on every card.

---

## 7. The progress bar

Two independent causes, both fixed.

**In the clip.** libx264's default GOP is 250 frames, so a 30-second render held
**three** keyframes — measured directly on a Round-6 clip. Dragging the scrubber
could only ever land on 0, 10 or 20 seconds. The renderer now asks for one
keyframe per second (`-g`, `-keyint_min`, `-sc_threshold 0`); the same clip now
has 30.

**In the server.** A player seeks by asking for a byte range. Python's
`http.server` answers every request with `200` and the whole file — it has never
implemented `Range` — and browsers read that as "not seekable", give up, and
restore the previous position. That is the snap-back.

`scripts/serve_gallery.py` is a drop-in replacement that answers `206 Partial
Content`, verified with mid-file, suffix and multi-request reads, on HTTP/1.1 so
a scrubbing burst reuses one connection. It binds to 127.0.0.1 only; participant
media does not leave the cluster.

```bash
$VIBES scripts/serve_gallery.py artifacts/private_review_vendors_r7 --port 8000
# then forward it:  ssh -N -L 8000:localhost:8000 <this-host>
```

Opening the HTML directly over a `file://` path or through a plain
`python -m http.server` will still snap back — the keyframe fix helps but does
not substitute for range support.

---

## 8. Gallery layout, now standard

Per your note, every gallery from here uses the briefing layout: the pipeline and
what it did to this sample at the top, a table of contents, a rule and a heading
between sections, the signals for each clip visible on the card, no free-text
notes, and FM2-flagged clips positioned on the longest continuous untrusted
stretch. `src/seamless_curation/briefing_gallery.py` is the single implementation;
the reviewer-gallery path with notes and rubrics still exists but is no longer
what new galleries are built from.

Cards now also carry a verdict badge, and the signal rows are configurable per
gallery, which is how this round shows raster, repair, shin direction and ankle
visibility alongside the three detector numbers.

---

## 9. The gallery

`artifacts/private_review_vendors_r7/` — **270 clips, 30 s each, 20 sections**,
236 participants over 240 sessions. Serve it with
`scripts/serve_gallery.py` (§7); a `file://` path will not scrub.

**Five ASK sections**, 68 clips, at the top:

| section | clips | what it asks |
|---|---|---|
| ASK 1 · V03 posture | 42 | sitting or standing, 6 clips in each of 7 bands from 0.20 to 0.80 |
| ASK 2 · V01's other rasters | 8 | are 1012×1920 and 1920×1080 usable once un-stretched |
| ASK 3 · V03's four rasters | 16 | 640×480, 1080×960, 3840×2160, 2180×3840 |
| ASK 4 · V01 2160×2160 | 6 | picture repaired, pose not — is that what you see |
| ASK 5 · drifting timebase | 6 | do the keypoints stay on the person now |

**Fifteen sample sections**, 202 clips, from a fresh random draw at seed 20261005,
disjoint from Round 6: per vendor, one section each for pass, FM1 alone, FM2
alone, FM3 alone, and more than one check. FM2 and multi-flag clips are
positioned on the longest continuous untrusted stretch — verified, 76 of 76 that
have one.

The rare rasters hold 20 to 102 files each, so a 1,500-per-vendor draw misses
them; a companion scan covers *every* file of all six and is used only to
populate ASK 2 and ASK 3. It is excluded from the rates below.

What the fresh sample says, at unchanged thresholds:

| vendor | files | FM1 | FM2 | FM3 | survives | projected dyad-hours |
|---|---|---|---|---|---|---|
| V01 | 1,495 | 43.61% | 70.17% | 9.30% | **14.38%** | 47 |
| V02 | 1,500 | 7.47% | 78.33% | 5.73% | **16.33%** | 124 |
| V03 | 1,500 | 16.73% | 80.33% | 6.27% | **13.00%** | 186 |

**358 dyad-hours** across the three, against V00's 419. Within a point or two of
Round 6 on every cell, which is what an independent draw at unchanged cuts should
give, and worth reading as *what the current pipeline says* rather than as a
verdict — FM1 accounts for most of V01's rejections and is measuring the
anamorphic distortion, not posture.

---

## 10. What I did not do

- **Did not re-tune FM1 for V03.** It needs labels on both sides of the cut, and
  guessing would bake in an unmeasured number. ASK 1.
- **Did not change FM1's verdict on legs-out-of-frame files.** You said the
  current behaviour is likely fine; I made the reason visible instead.
- **Did not gate FM1 on frame shape.** Round 6 recommended it. With the pixel
  aspect now understood, the square-raster collapse is explained rather than
  merely observed, and the right response is a decision about whether to keep
  those files at all (ASK 4) rather than a threshold change.
- **Did not re-fit any SMPL-H.** That is a new HMR run over the corpus, not a
  settings change, and §1.4 is the argument for when it would be worth it.
- **Did not re-render the V00 briefing gallery** your PI is reviewing. The
  repairs are all no-ops on V00 — square pixels, upright, consistent timebase,
  no empty WAVs — so the only change would be the denser keyframes.
