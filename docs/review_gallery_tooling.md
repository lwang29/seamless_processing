# Phase-0 private review gallery tooling

This tooling renders 30-second review clips with four synchronized regions:

1. released COCO-WholeBody keypoints and bounding box on video, with independent
   `smplh`, box, and optional movement mask dots;
2. acceptance-tested projected SMPL-H joints on video;
3. fixed-scale, root-relative SMPL-H joints in front and side orthographic views;
4. released-WAV waveform, own/partner VAD shading, and a moving playhead.

The renderer does not independently interpret the camera. It consumes a
`ReviewJointProvider` returning native-pixel projected joints and pelvis-subtracted
3D joints in metres. M-4 accepted the configured adapter using neutral SMPL-H,
16 all-zero betas, full axis-angle hands, and `flat_hand_mean=True`. Projection
uses the accepted full-frame HMR-style camera. The auxiliary face and toe/heel
landmarks are not drawn; M-4 found structured foot-extra outliers. If the
provider is absent or fails, Panels B and C visibly say unavailable rather than
showing a guessed skeleton.

## Panel layout

Round 4 renders three visual panels plus a whole-file filmstrip and the audio
strip:

| region | content | why it exists |
|---|---|---|
| filmstrip | 10 full-frame thumbnails spanning the whole recording, timestamped, with a red marker for the clip's position | Most of the reviewer's vocabulary is file-level — "hands static for the entire conversation", "framing consistent" — and one excerpt cannot answer it. |
| A | full frame with released 2D keypoints, box, and S/B/M mask dots | Context: is the participant in frame, is the box tracking, do the masks flip |
| B | M-4-accepted SMPL-H FK projected into the same frames | Does the fitted body match the person |
| C | root-relative 3D, fixed-scale front and side | Is the 3D motion plausible independent of the camera. The side view stands the torso axis vertical (`upright_side_view`); see below. |
| strip | waveform with own/partner VAD and a playhead | Speech context, and desync is audible because audio is muxed |

`duration_s` and `filmstrip_thumbnails` are set under `clip:` in the review
config; `filmstrip_thumbnails: 0` disables the band.

### What the S/B/M dots mean

The three dots in the top-right of Panel A are the **released per-frame validity
masks**, read straight out of the NPZ — they are the dataset's own opinion of
each frame, not anything this project computed:

| dot | NPZ key | asserts |
|---|---|---|
| **S** | `smplh:is_valid` | the SMPL-H fit was trusted on this frame |
| **B** | `boxes_and_keypoints:is_valid_box` | a person box and its keypoints were tracked on this frame |
| **M** | `movement:is_valid` | the derived movement/expression features are present for this frame |

**Green** means the mask is true for the frame on screen, **red** means false,
and **grey** means the file carries no such mask at all (`movement:is_valid` is
optional and absent from some files).

A red **B** is the most serious: on every box-invalid frame audited, the box and
all 133 keypoints were exactly zero and the SMPL-H translation held a ~1e12–1e14
sentinel, with the video showing the participant out of frame or a cut to black.
A red **M** is the least alarming: it arrives in exact 60-frame blocks with the
features zero-filled and no visible change in the video.

A red **S** is subtler than it looks, and Round 5 measured what it actually
means. The body is *fine* on those frames — paired within 2,481 files, the median
2D fit error against the released keypoints is 0.0792 shoulder widths on valid
frames and 0.0799 on invalid ones. What changes is the **hands**: on invalid
frames about 40% of left- and right-hand pose vectors are bit-identical to the
previous frame (0% on valid frames) and median per-frame hand motion falls to
0.33× and 0.03×, while body pose, global orient and translation are unaffected.
`smplh:is_valid` is therefore, in effect, a hand-pose flag: the pipeline holds the
previous hand pose when it does not trust the hand fit. A red S means distrust the
hands in Panels B and C specifically — which is why FM2 now gates on it.

### Filmstrip endpoints

The first thumbnail is frame 0 and the last is the final frame, with the rest
evenly spaced between them (`filmstrip_frame_targets`). The pre-Round-4 scheme
sampled bin *centres*, so neither end of the recording ever appeared — exactly
where a framing change is most likely. Round 5 cut the count from 12 to 10,
because at 12 the thumbnails were too small to read.

Seeking to the last frame needs care. On V00_S0180_I00000482_P0047 (`nb_frames`
9420, 30 fps, duration 314.000 s) `ffmpeg -ss 313.967` — that is `(n-1)/fps` —
returns **no frame at all**, while `-ss 313.933` returns one; `nb_frames`
over-counts the seekable frames by one. `_grab_frame_stepping_back` therefore
retreats a frame at a time, up to four attempts, and reports the timestamp that
actually worked so the printed label stays honest. Without it the last thumbnail
of every filmstrip would be a permanent grey `?`.

An unreadable seek is still drawn as an explicit purple gap with a `?`, not
skipped — a missing thumbnail is itself information about the file.

### Retired: Panel A′

Round 2 added a fixed upper-body-and-hands crop as a fourth panel, because a
portrait raster gives hands only ~216 px. The reviewer dropped it in Round 4, and
it is gone rather than defaulted off: `upper_body_crop_box` is deleted and
`RenderSettings` no longer has `upper_body_crop`, `crop_aspect`, or
`crop_pad_frac`. Configs written before Round 4 still carry those keys, so
`normalize_review_config` drops them (`_RETIRED_RENDER_KEYS`) instead of raising
on an argument that no longer exists.

## Overlay thickness and the panel downscale

Overlays are drawn on the native-resolution frame, and the panel is then
downscaled to fit the 480-px-tall canvas — a factor of 10 for a 2160×3840 V03
file. A thickness chosen in native pixels therefore lands well below one output
pixel and `INTER_AREA` averages the skeleton away; measured effective widths were
0.30–0.53 px across the four dominant rasters before this was fixed.
`overlay_thickness` now pre-compensates the known `panel_display_scale` to
target about two output pixels, and a test asserts the compensated value survives
the downscale for every dominant raster. `renderer_version` is now 5 (2 fixed the
thickness, 3 added the crop panel and filmstrip, 4 removed the crop panel, moved
the filmstrip to both endpoints and went to 30-second clips, 5 cut the filmstrip
to 10 thumbnails); clips rendered under an earlier version do not match the
fingerprint and are re-rendered.

Portrait rasters still get a narrow panel (216 px for 2160×3840). Individual
finger joints are consequently small. Round 2 addressed this with a cropped
Panel A′; Round 4 retired it, so the remaining route to hand detail is the
full-recording download beside each clip.

## Manifest contract

Required CSV columns are `review_item_id`, `clip_id`, `file_id`,
`source_relbase`, `sample_group`, and `start_frame`. Optional columns include
`partner_source_relbase`, `vendor`, `label`, `split`, `activity_type`,
`selection_reason`, `signals_json`, and `render_policy`.

Duplicate review items use distinct `review_item_id` values but may share a
`clip_id`, source, and start frame. The HTML stores notes by `review_item_id` and
does not identify duplicates to the reviewer. `render_policy=metadata_only`
keeps an unrenderable placeholder bundle in the gallery without inventing media.

`signals_json` is a JSON object containing measured signals. Alternatively,
columns prefixed with `signal__` are displayed as signals. Sample selection is a
separate step; this renderer never selects examples.

## Private output and review notes

The configured output directory is forced to mode `0700`; MP4, JSON, and HTML
outputs are forced to `0600`. It is beneath Git-ignored `artifacts/`. Audio is
the released participant WAV, clipped at `start_frame / avg_frame_rate`, encoded
as mono AAC, and muxed with the H.264 panels. The strip uses that same WAV.
Partner VAD is read from the explicitly supplied paired JSON path; absence is
recorded rather than inferred by scanning the corpus.

Without `--rubric` the HTML is the Pass-1 gallery: unrestricted free text and
**no rating machinery at all**. A test asserts the page contains no radio inputs,
no rubric fieldset, and not even the substring `rating`, because offering
categories is what an exploratory pass exists to avoid. Notes persist in browser
local storage and export as JSON with identifiers and text, never media.

With `--rubric <path>` the same clips render the Pass-2 structured gallery. The
rubric is validated, not trusted: it is refused unless `status: approved` and
`derived_from` names the Pass-1 notes it came from, and item ids must be unique
snake_case with `1 <= min < max <= 9`. **No rubric content exists in this
repository.** `tests/fixtures/rubric_test_only.yaml` uses deliberately
meaningless item names so that reading it cannot seed a reviewer's vocabulary.

The Pass-2 export uses schema `structured_ratings_v1` and is stamped with the
rubric hash and its `derived_from` provenance, so ratings can never be silently
attributed to the wrong rubric.

Each playable card carries three controls beside the player:

| control | target |
|---|---|
| **Download full video** (primary, with size) | the whole source recording, so a file-level judgement can be checked end to end |
| **30-s panel clip** | the rendered clip with the review panels |
| **Open full size** | the same panel clip, on its own |

The middle button's label comes from the clip's own recorded `duration_s`
(`_panel_clip_label`) rather than a hard-coded string, so it cannot promise ten
seconds and hand over thirty; a record with no recorded duration reads just
"panel clip".

The full recordings are **symlinked, not copied**, into a mode-`0700` `source/`
subdirectory of the gallery root: the Round-5 gallery's 425 links cost 42 KB of
disk against 15 GB of targets. A link under the gallery root rather than a `../..` path into the source
tree keeps the page self-contained, so it works both as a `file://` URL and under
a static server rooted at the gallery directory.

`link_source_media` never chmods through a symlink — that would follow the link
and try to modify the read-only source file. A test asserts the source keeps its
inode, mode, and bytes. Records whose source is absent gain no link, so the page
never offers one that does not resolve. Set `outputs.link_source_media: false` to
turn the whole thing off.

`--gallery-name` writes `<name>.html` and `<name>_render_results.json`, which is
how a Pass-1 subset gallery reuses already-rendered clips without clobbering the
full index. `--title` overrides the heading.

## Bounded execution

Run from an allocated compute node with the existing environment:

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
/simurgh/group/lw29/conda/envs/ViBES/bin/python scripts/render_review_gallery.py \
  --config configs/review_gallery_dryrun.yaml --limit 1 --strict
```

Use `--index N` for one array item. Matching clip/config/provider fingerprints
are reused; `--overwrite` forces a rerender. Each indexed task writes a unique,
atomic result JSON and does not touch the shared HTML or aggregate manifest.
Submit `slurm/render_review_gallery_array.sbatch` with an explicit bounded array,
for example `sbatch --array=0-299%8 ...`. After all tasks finish, one
`--html-only` invocation builds the index and marks any missing task `pending`.
Errors are represented as gallery cards unless `--strict` is used.


## The briefing gallery

`src/seamless_curation/briefing_gallery.py` is a second, separate page builder,
not a mode of the reviewer gallery. The two have opposite jobs and mixing them
would compromise both.

| | reviewer gallery | briefing gallery |
|---|---|---|
| purpose | collect judgements | explain finished work |
| per clip | free-text notes, every computed signal | six curated rows, no input |
| detector opinions | played down, to avoid leading the reviewer | stated up front, with the reasoning |
| layout | one flat grid | five sections, dividers, sticky headings, table of contents |
| statistics | none on the page | flag rates, pass rate, projected dyad-hours |

A test asserts the briefing page contains no `<textarea>`, `<input>`, `<form>`,
radio, `localStorage`, or export control. Someone who is not being asked for
input should not be shown a box that invites it.

`SIGNAL_ROWS` is the whole per-clip signal set, six rows chosen because each
makes one verdict legible. Everything that needed a paragraph to interpret was
cut; the page has to be readable cold. A file with no untrusted frames reads
"none" rather than "0.0 s", because zero here is an absence and not a rounded
measurement.

The module computes no statistics and owns no thresholds — every number is passed
in from the analysis of the sample the page shows, so the prose cannot drift away
from the measurements. `scripts/build_v00_briefing_gallery.py` reads them from
`analysis_summary.json` and `flag_rates_by_label.csv`.


## Panel C and the camera tilt

Panel C plots **camera-frame** coordinates, so a camera that is not level draws an
upright participant leaning. V00 was shot from above eye level angled down, and
over 440 standing participants the median pelvis-to-neck axis sits **13.1 degrees**
out of the image plane, head toward the camera — a per-session property
(within-session SD 1.88 degrees against 5.10 degrees between sessions), which is
the signature of a fixed rig rather than of posture.

`upright_joints` rotates about the camera x-axis so the torso axis stands
vertical before the side view is drawn. It is a **rigid rotation**: every joint
angle and every pairwise distance is preserved, and a test asserts it. That
matters, because the residual knee bend the corrected view still shows is real —
it lives in the released `body_pose`, is held about 59 degrees from straight, and
is required to fit the released 2D keypoints. See
`reports/06_smplh_pose_geometry.md`.

Set `clip.upright_side_view: false` to draw the raw camera frame instead. The
panel labels itself "side: z/y upright" or "side: z/y" so the two cannot be
confused.
