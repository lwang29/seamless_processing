# The v0 measurement rounds (2026-08-04 to 2026-08-09)

Sixteen reports from fifteen rounds of measuring the corpus and building
file-level failure-mode detectors. **None of them describes the current
pipeline**, which is in [`../17_cospeech_gesture.md`](../17_cospeech_gesture.md).
They are kept because most of the numbers the current pipeline relies on were
established here, and because two of its design choices are only defensible if
you can see the evidence that the alternatives were tried and failed.

The code they describe has been removed. The reports have not been edited, so
they still refer to modules and scripts that no longer exist; treat any command
line in them as history.

## What is still load-bearing

| finding | where | why it still matters |
|---|---|---|
| Every jitter and acceleration variant is confounded with gesture activity, up to rho 0.903 over 135 candidate signals — including the high-pass form built specifically to suppress smooth motion | `02_measurements.md` §on quality-vs-activity | This is why the current pipeline has **no jitter-magnitude gate**. Any such gate is arithmetically a gate on how much the participant gestured. |
| Naive axis-angle deltas exceed 100 rad/s on 43.5% of dev spans, up to 813x the geodesic p99 | `02_measurements.md` | All angular velocity is composed, never subtracted. |
| `smplh:is_valid == False` freezes the *hand* pose, not the body: ~40% of adjacent hand-pose vectors bit-identical against exactly 0% on valid spans | `01_recon.md`, `02_measurements.md` | `hand_frozen_frac` exists because of this; it is the measured damage behind an invalid frame. |
| An anamorphic file's released SMPL-H cannot be repaired — the isotropic fitted camera forced the stretch into the pose | `08_vendor_repairs.md` §1.4 | V01's 2160x2160 and 1920x1080 rasters are excluded from the population. |
| Quarter-turned V03 rasters and pixel-aspect corrections are exact image-plane transforms | `08_vendor_repairs.md` | `media_repair.py` survives unchanged. |
| The camera is angled down; upright participants read as leaning, and the ~59-degree knee bend is real and confined to the legs | `06_smplh_pose_geometry.md` | Do not read a lean in a raw camera-frame view as a posture defect. |
| M-4 reprojection selected neutral SMPL-H, betas=0, `use_pca=False`, `flat_hand_mean=True` | `02_measurements.md` §M-4 | The whole pipeline's SMPL-H convention. |
| Released 2D is COCO-WholeBody-133; box-invalid frames are zero-filled wholesale; confidence is bimodal at ~0.9 or exactly 0 | `01_recon.md`, `02_measurements.md` | `kp_conf_p10` is in practice a box-validity gate, not a confidence filter. |
| 627 participant ids collide across vendors | `02_measurements.md` §M-1 | Identity is `(vendor, participant_id)` everywhere. |
| Naturalistic recordings are 2.3x more likely than improvised to have static hands for over half the recording | `02_review_findings.md` §10 | Context for the accept rates by condition. |

## What was superseded, and why

- **FM1, seated posture** (`10`–`13`). Five rounds of work, retired outright:
  ViBES trains the upper body, and the PI ruled that seated posture and legs out
  of frame are not grounds for rejection.
- **FM2, whole-file SMPL-H validity** (`05`). Requiring a valid fit on every
  frame of the file rejected 62.6% of V00 — more than every other check
  combined. Replaced by a per-window allowance plus `hand_frozen_frac`.
- **FM3, static hands** (`04`, `05`). The direct ancestor of the current gesture
  measure and the reason it exists. It is whole-file, 2D-only, never reads the
  VAD, needs *both* wrists parked simultaneously, and anchors on a per-file
  median — so a participant who gestures while speaking and rests at their hips
  the other 75% of the time is flagged, while one whose hands never leave their
  waist is not. Its two constants, 0.10 shoulder widths and 0.75, were chosen a
  priori and never had a confusion matrix behind them.
- **FM4, dead audio** (`09`, `14`, `15`). Superseded structurally rather than
  refuted: a file with no voice has an empty released VAD, so it fails the
  current pipeline's speech gate before any audio measure is needed. The
  envelope-dynamics result in `15` is still the right way to detect a dead track
  if one is ever needed again.
- **The review galleries** (`04`–`16`). Read-only briefing pages that by design
  collected nothing, and before them a note-taking page whose notes lived in
  browser `localStorage` under a key that changed on every re-render.

## Reading order, if you need the whole story

`01_recon.md` → `02_measurements.md` → `02_review_findings.md` →
`04`/`05_failure_mode_detection.md` → `06_smplh_pose_geometry.md` →
`07_other_vendors.md` → `08_vendor_repairs.md` → `09`–`15` (threshold
corrections, one round each) → `16_pi_briefing.md`.
