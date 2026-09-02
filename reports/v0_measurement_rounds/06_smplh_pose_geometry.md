# Why upright participants read as leaning with bent knees

You noticed that participants standing straight in the video show, in Panel C's
side view, a forward-tilted body with bent knees and feet off the flat, and
suspected the camera. **You are right about the cause, and the two symptoms
split cleanly — one is a rendering choice we can fix in a line, the other is real
and lives in the released data.**

Short version:

| symptom | verdict | fixable by a setting? |
|---|---|---|
| whole body tilted forward | **rendering.** Panel C plots camera-frame coordinates and V00's camera is angled down. | **Yes — done.** One rotation. |
| knees bent, feet not flat | **real.** It is in the released `body_pose`; no rendering change can touch it. | No. It would need a re-fit. |
| does it hurt gesture training? | **Probably not**, because it is confined to the legs, which are not in the pelvis-to-hand chain. One caveat in §5. |  |

Measured on **440 standing participants that pass all three checks**, plus a
per-joint reading of 480 files. Commands: `scripts/probe_smplh_tilt.py`,
`scripts/probe_smplh_joint_bend.py`, `scripts/probe_smplh_side_views.py`.

---

## 1. The tilt is the camera, and Panel C was showing it faithfully

Panel C plots **camera-frame** coordinates — the body as the camera sees it, with
no gravity alignment. So a camera that is not level draws an upright person
leaning. That is a display convention, not a pose error.

Measuring the pelvis-to-neck axis against the image plane, where 0° is upright
under a level camera:

| | value |
|---|---|
| median body-axis pitch | **−13.1°** (head toward the camera) |
| interquartile range | −17.6° to −9.0° |
| p5 / p95 | −23.7° / −5.2° |

Negative means the head is nearer the camera than the pelvis, which is what a
camera **above eye level angled down** produces for a standing person. Exactly
your hypothesis.

**It is a property of the rig, not of the person.** Within a recording session
the pitch varies by 1.88°; between sessions by 5.10° — a 2.7× ratio, over 109
sessions with two or more measured files. Within a single file it wanders 2.38°,
which is most of the within-session figure. A camera bolted in place for a
session and re-rigged between sessions is exactly this signature.

For context, the camera the fit assumes is HMR's fixed weak-perspective one:
focal length **37,500 px**, vertical field of view **2.93°**, which places the
participant at a fitted depth of **39.8 m**. A real studio camera is a few metres
away with a 50–70° field of view. That mismatch matters in §3.

### Fixed

`upright_side_view` now stands the torso axis vertical before drawing Panel C's
side view (`upright_joints` in the renderer, default on, `renderer_version` 6).
It is a rigid rotation, so **every joint angle is unchanged** — a test asserts
that the knee angle and all pairwise distances survive it, because the point of
the correction is to remove the camera without hiding the pose.

`outputs/session2/side_views/side_view_comparison.png` shows four files as video,
Panel C today, tilt removed, and tilt removed with the knees forced straight.
With the tilt gone the torso stands vertical; the legs stay bent.

**I have not re-rendered any existing gallery**, including the one your PI has.
Changing an artifact underneath a review in progress seemed worse than a stale
side view. One command applies it whenever you want:

```bash
sbatch slurm/render_v00_briefing_array.sbatch && \
  $VIBES scripts/build_v00_briefing_gallery.py
```

---

## 2. The knee bend is real

A joint angle is invariant to every global rotation, so no plotting choice can
create or remove one. Reading the released `body_pose` directly — no forward
kinematics, no camera, nothing that could be a projection artefact — and
measuring how far each joint is held from the SMPL rest pose:

| joint | held from rest | moves within a file | varies across files |
|---|---|---|---|
| **L / R knee** | **58.7° / 59.8°** | ±5.2° | 18.1° |
| L / R hip | 23.1° / 23.5° | ±3.2° | 7.5° |
| L / R ankle | 12.5° / 12.2° | ±1.0° | — |
| spine1 | 20.2° | ±2.3° | 4.6° |
| **spine2** | **2.2°** | ±0.8° | 0.7° |
| **spine3** | **2.6°** | ±1.1° | 1.6° |

By region: legs **17.8°** from rest, torso **3.3°**, head and neck 10.0°.

So the knees are held about **59° from straight** — a visible half-squat, not the
soft knees of ordinary standing. The feet agree: the foot sits at a median
**131.5°** from the body axis where 90° would be flat under an upright body.
Both are exactly what you saw.

Note the arms read 45° from rest, but that is not distortion: SMPL's rest pose is
a T-pose, so arms hanging at the sides are legitimately tens of degrees from it.

---

## 3. But the bend is *required* to fit the 2D data — which points at the camera

The obvious next question is whether the bend is free-floating depth noise, since
monocular fitting is famously unconstrained in depth. It is not.

Forcing both knees straight and re-projecting, on all 440 files:

| | lower-body reprojection error |
|---|---|
| as released | 0.0875 shoulder widths |
| knees forced straight | **0.1403** shoulder widths |

**+61% worse, in 439 of 440 files** (Wilcoxon p = 8×10⁻⁷⁴). Upper-body error is
untouched at 0.0707 either way, as it must be. So the bend is doing real work
holding the fitted knees and ankles onto the released 2D keypoints.

That combination — the bend is anatomically implausible, the video shows people
standing straight, and yet the bend is what the 2D data demands — points at the
camera model rather than at the pose optimiser. Under a real camera a few metres
away and angled down, a standing person's legs foreshorten in the image. The
assumed camera is near-orthographic with **no pitch parameter at all**, so it
cannot represent that foreshortening. The only remaining way to put the knees and
ankles where the image shows them is to physically bend the legs in 3D.

Two pieces of supporting evidence:

* the forward pitch of the upper body relative to the pelvis (spine1) correlates
  with the camera-frame tilt at Spearman **−0.605**;
* spine1, like the camera tilt, varies more between sessions (3.97°) than within
  them (1.47°) — it tracks the rig, not the person.

**Stated as inference, not measurement.** I have shown the bend is demanded by
the 2D keypoints under this camera, and that the postural distortion tracks the
recording setup. I have *not* shown that a correctly calibrated camera would make
the bend disappear; that would need a re-fit with a proper perspective model,
which is a new HMR run over the corpus rather than a settings change. The one
correlation that does not fit cleanly is pitch against knee bend, at −0.385 —
moderate, and in the opposite direction to the simplest version of the story, so
the mechanism is probably not a single scalar effect.

---

## 4. It cannot be corrected by a setting

The tilt can (§1). The bend cannot: it is baked into the released `body_pose`
arrays, and §3 shows we cannot simply straighten the legs, because doing so
provably breaks agreement with the released 2D keypoints. The options are to
accept it, to exclude the legs from the training target, or to re-fit the corpus
with a calibrated camera. Only the third actually removes it, and it is a
substantial piece of work.

---

## 5. Whether it matters for training

Mostly not, for one structural reason: **the legs are not in the kinematic chain
from the pelvis to the hands.** A co-speech gesture model trained on upper body
and hands never reads the knee or hip rotations, so a 59° error there cannot
reach its output.

What is in that chain, and what state it is in:

| link | held from rest | across files | verdict |
|---|---|---|---|
| spine1 | 20.2° | ±4.6° | a static per-recording pitch offset |
| spine2 | 2.2° | ±0.7° | clean |
| spine3 | 2.6° | ±1.6° | clean |
| neck | 15.3° | ±4.3° | static offset |
| collars, shoulders | 28°, 57° | ±5° | mostly the T-pose rest offset |
| elbows, wrists | — | ±59°, ±48° | this is the gesturing, i.e. the signal |

So the torso interior is essentially undistorted, and the one contaminated link,
spine1, is a **static** offset within a recording (±2.3°) that varies by only
**±4.6°** across recordings. That is a small nuisance variable, not noise. The
global camera tilt is absorbed entirely by `global_orient`, which any
root-canonicalising pipeline discards.

Three honest caveats:

1. **Depth is unverified.** Reprojection error is a 2D measure, so I can confirm
   the upper body lands on the image (0.0707 shoulder widths, about 25 px) but
   not that its *depth* is right. For 3D gesture output, arm depth is the thing
   that matters most and this investigation cannot rule on it. It is the same
   blind spot flagged in the Round-5 report.
2. **If you ever want full-body output**, including legs or foot contact, the leg
   pose is not trustworthy and this becomes a blocking problem rather than a
   cosmetic one.
3. **Root translation inherits the camera.** The fitted depth of 39.8 m is not
   physical, so absolute root trajectory is not metric. Session 2 already
   established this and renamed the affected column accordingly; it is unchanged
   by anything here.

---

## 6. Recommendation

1. **Take the Panel C fix** (already implemented, not yet applied to any
   rendered gallery). It makes the side view answer the question it appears to
   answer, and it hides nothing.
2. **Train on upper body and hands in a root-relative frame**, which was the
   plan anyway. The leg distortion then never enters the model.
3. **Do not use the leg pose for anything** — not foot contact, not stance, not
   full-body reconstruction — without a re-fit.
4. **Treat arm depth as an open risk.** If the model produces 3D gesture and the
   depth looks wrong later, this is the first place to look, and the check would
   be a multi-view or depth-sensor comparison that the release does not support.
