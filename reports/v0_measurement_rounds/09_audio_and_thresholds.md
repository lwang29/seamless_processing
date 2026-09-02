# Round 8: the audio problem, and the first round where thresholds moved

Your Round-7 answers were decisive enough to change the pipeline rather than just
annotate it. Four things moved:

| | change | from |
|---|---|---|
| **FM0** *(new)* | reject V01's anamorphic rasters, V03's room-camera rasters, and files whose annotations do not match their video | your ASK 2/3/4/5 answers |
| **FM1** | per-vendor: 0.43 on V00, **0.54 on V03**, **off for V01 and V02** | your 42 posture labels, plus the 16 from Round 6 |
| **FM4** *(new)* | reject a recording whose audio carries no voice | your two "completely silent" clips and two of the 21 glitchy ones |
| repair layer | crop the black columns off 1080×960 and 2180×3840 | your ASK 3 answer, confirmed by measurement |

Net effect: **313 projected dyad-hours** across V01–V03, down from 358. Most of
that is V03, whose pass rate goes 13.0% → 9.1% because the recall-biased FM1 cut
is doing what you asked it to.

New gallery: `artifacts/private_review_vendors_r8/` — 204 clips, 25 sections.
**Serve it with `scripts/serve_gallery.py`** (§1).

---

## 1. The progress bar, again — and what I got wrong the first time

You were right that it was still broken. My Round-7 fix addressed the clip
(three keyframes per 30 s → thirty) and the server (`http.server` never
implemented HTTP `Range`, so it answers a seek request with `200` and the whole
file, which browsers read as "not seekable"). Both were real. Neither helps if
you are not going through that server — and an editor preview pane serves its own
local resources, so it hits exactly the same wall.

So this round adds the layer that works on **any** transport: once a clip is
*entirely* in the browser's buffer, seeking needs no request at all. Each video
is upgraded from metadata-only to a full download as it scrolls into view, and a
pill next to it says when scrubbing will work:

> `buffering 62%` → `fully buffered · scrubbing works`

Only what you are looking at downloads, so the page still opens without fetching
a gigabyte. Clips average 4.1 MB, so a card is usually ready before you have
finished reading its signals.

There is also now a banner at the top of the page that runs a range request
against the first clip and tells you what the transport did:

> ⚠ This transport answered a range request with **200** instead of 206, so
> seeking will only work once a clip has fully buffered…

If that banner is green and scrubbing is still broken, the problem is somewhere I
have not looked, and the banner tells us that in one glance instead of a round
trip. For instant scrubbing:

```bash
$VIBES scripts/serve_gallery.py artifacts/private_review_vendors_r8 --port 8000
ssh -N -L 8000:localhost:8000 <cluster-host>   # then open http://localhost:8000/
```

in a normal browser rather than an editor preview.

---

## 2. The glitchy audio: three hypotheses, one survivor

Your 21 clips are real and I can now say what they are, but the honest answer has
a limit in it, so §2.3 matters as much as §2.2.

### 2.1 What it is *not*

Measured across 2,793 files — a random draw per vendor plus every file of the six
unusual rasters:

| hypothesis | result |
|---|---|
| clipping, peak level, DC offset | overlap completely with good recordings |
| digital dropouts (exact-zero runs mid-speech) | **zero** in every flagged file |
| brick-wall spectral cutoff, aliasing | flagged files have *more* high-frequency energy, not less |
| fragmentation — speech chopped into short bursts | median burst 0.17 s in flagged files and 0.17 s in unflagged ones |

The fragmentation one is worth dwelling on because the waveforms look chopped and
it is the obvious reading. It is wrong: every dyadic recording looks chopped,
because each participant is silent about half the time. Identical statistics on
both sides.

### 2.2 What it is: the microphone is hearing the wrong person

Comparing a track's energy during **its own** speaker's turns against its energy
during the **partner's** turns, using the released VAD:

| raster | files | own voice above partner's |
|---|---|---|
| V01 1012×1920 | 102 | 12.8 dB |
| V03 2180×3840 | 20 | 11.1 dB |
| V03 2160×3840 | 291 | 9.2 dB |
| V03 1080×960 | 28 | 8.0 dB |
| V01 1080×1920 | 178 | 7.7 dB |
| V00 1080×1920 | 300 | 7.3 dB |
| **V03 640×480** | **55** | **3.8 dB** |
| **V03 3840×2160** | **1,318** | **2.2 dB** |

A close-worn microphone hears its wearer far better than the other person. At
2 dB both speakers arrive at the same level, which is what a room or camera
microphone does — and V03's 3840×2160 and 640×480 files are exactly the wide
room-camera framings. Far-field pickup, room reverb, and automatic gain riding
over the top of two overlapping voices. That is what you heard.

It also explains the correlation you noticed with abnormal aspect ratios: it is
not the aspect ratio, it is that the odd rasters *are* the room-camera rig.

### 2.3 The limit: it does not separate file by file

The per-raster medians are clean. The per-file distributions are not:

| cut | catches, of 20 labelled glitchy | but also takes |
|---|---|---|
| < 6 dB | 13 | 31.5% of V00/V02 |
| < 8 dB | 16 | 47.3% of V00/V02 |
| < 10 dB | 17 | 59.2% of V00/V02 |

I am not going to ship a detector that rejects half of V00. So voice isolation is
recorded as a **signal** on every card and **ASK 2** in the gallery collects the
labels that would turn it into a detector — a ladder across the range, spread over
all three vendors so the answer is about the measure and not the vendor.

Meanwhile, the raster rule you asked for (§3) removes 1,373 of the 1,396
room-camera files anyway, which is most of the problem.

### 2.4 What did become a detector: FM4, dead audio

Two of your 21 glitchy clips and both of your "completely silent" ones are not
glitchy at all — they carry **no voice**. Speech level −63 to −88 dB against a
corpus median near −23, with a flat long-term spectrum: a noise floor, not a
person.

**FM4 = speech level < −55 dB *and* spectral flatness > 0.05.** Both halves are
required — a genuinely quiet recording is not broken, and a flat spectrum in a
loud file is a fan.

| | result |
|---|---|
| the four files you confirmed dead | **all four caught** |
| the 18 that are audible but unpleasant | **none caught** — it is specific, not a rubber stamp |
| V00 control | **0 of 300** |
| fires on | V01 3.80%, V02 2.13%, V03 0.87% |

Note it catches the two 2160×2160 silent clips *independently* of the raster
rule, so it would have found them even if V01's square raster were being kept.

---

## 3. FM0: the exclusions you decided

Your ASK answers were decisions, so they are now rules rather than notes. Each
carries the reason it was taken:

| raster | files | dyad-hours | why |
|---|---|---|---|
| V01 2160×2160 | 5,002 | 139.0 | picture correctable, released SMPL-H not — you saw the Panel B/C fit |
| V01 1920×1080 | 60 | 1.5 | same, worse; you called the skeleton "glitchy and a little misaligned" |
| V03 640×480 | 60 | 2.0 | far-field audio, and 0.3 MP besides |
| V03 3840×2160 | 1,336 | 41.8 | far-field audio on all but one sampled file |
| — timebase mismatch | ~70 | — | you found the SMPL-H "completely incorrect and looks to be from a different video" on two |

**Kept:** V01 1012×1920 (102 files) as you judged, and V03 1080×960 and
2180×3840 with the black columns cropped.

On 3840×2160 your words were "leaning towards filtering them all out", which is a
lean rather than a decision — and `V03_S0959_I00000441_P1813` did have acceptable
audio. It is 41.8 dyad-hours and one line in `configs/vendors_r8_detect.yaml` to
reverse. Flagging it as the costliest of the five and the easiest to undo.

FM0 now rejects **44.6% of V01**, 0% of V02 and 3.9% of V03.

### The crop, confirmed

Measuring the brightest value each column ever reaches, over several frames per
file so a dark participant is not read as padding:

- **1080×960** → 270 black columns each side, content **540×960**. Exactly your
  reading.
- **2180×3840** → 20 black columns, on the **right only**, content **2160×3840**.

Both are now cropped before display. One thing worth knowing: **the padding was
costing nothing.** Both rasters fit as well as any ordinary one — shape residual
0.057 and 0.061 against a 0.048–0.074 normal band — so the crop makes the review
panel bigger and the aspect consistent, and changes no measurement.

---

## 4. FM1: per-vendor, and recall-biased on V03

### V03: 0.43 → 0.54

Your 42 labels plus Round 6's 16 give **58 V03 files** with a hand-verified
posture. They separate well, but not where the old cut was:

| | knee position along the torso |
|---|---|
| 36 seated | min 0.249, median **0.413**, max 0.630 |
| 22 standing | min **0.507**, median 0.589, max 0.724 |

V03's *seated* distribution sits about 0.11 above V00's (whose sitters read near
0.30) while its standing distribution is unchanged — so this is V03's seating,
not its camera. Sweeping the cut against all 58:

| cut | sitters caught | standing lost | V03 files rejected |
|---|---|---|---|
| 0.43 *(old)* | 21 / 36 | 0 / 22 | 22% |
| 0.50 | 28 / 36 | 2 / 22 | 25% |
| **0.54** | **33 / 36** | **4 / 22** | **36%** |
| 0.56 | 34 / 36 | 6 / 22 | 49% |
| 0.60 | 35 / 36 | 15 / 22 | 79% |

You said "better to filter out all or most sitting clips and accidentally filter
out a couple standing clips", so **0.54**: 92% of sitters for 18% of standing.
0.56 buys one more sitter and costs another 13% of the whole vendor, which is
where I stopped.

This is the change that costs the hours — V03 goes from 13.0% to 9.1% surviving.
**ASK 1** is drawn tight around 0.46–0.62 to check the new boundary; below and
above that the existing labels are unanimous.

Two smaller things. Your `V03_S0684_I00000388_P2446` (seated, FM1 silent) reads
0.61, so 0.54 still misses it — it is one of the three the new cut does not
catch. And both of your "neither" clips — one foot on a stool, and standing then
sitting — are already rejected by FM2 (SMPL-H valid on 73% and 83% of frames), so
they leave the corpus regardless.

### V01 and V02: off

You said all V01 and V02 FM1 flags were standing people. The measurements agree
and say why:

- **V01** fires on **0 of 1,379** square-pixel files. Its 43.6% rate was entirely
  the anamorphic rasters, which FM0 now removes. Nothing is left for FM1 to do.
- **V02** fires on 182 files, and **181 of those fire on the shin add-on**, not
  the knee measure — the ankles-out-of-frame mechanism from Round 7 that you
  confirmed as standing in all eleven cases you checked. Exactly one V02 file
  fires on knee position.

So FM1 is switched off for both. It is recorded as `retired_for_vendor` rather
than a clean pass, so no file reads as validated when nothing was asked of it.

---

## 5. Where this leaves the three vendors

Round-8 sample, 4,499 files, fresh seed disjoint from Rounds 6 and 7:

| vendor | files | FM0 | FM1 | FM2 | FM3 | FM4 | survives | dyad-hours |
|---|---|---|---|---|---|---|---|---|
| V01 | 1,499 | 44.56% | 0% | 69.71% | 8.54% | 3.80% | **15.41%** | 51 |
| V02 | 1,500 | 0% | 0% | 77.47% | 6.60% | 2.13% | **17.33%** | 132 |
| V03 | 1,500 | 3.87% | 30.07% | 82.00% | 6.13% | 0.87% | **9.07%** | 130 |

**313 dyad-hours**, against V00's 419. A file counts against every check it
trips, so the columns sum to more than the rejected share.

FM2 remains by far the largest cost everywhere and is untouched since Round 5, so
that column is directly comparable with every earlier round.

---

## 6. What I need from you

Four asks, and three of them are "did I put this in the right place" rather than
open questions.

| | clips | question |
|---|---|---|
| **ASK 1** | 32 | sitting or standing, tight around FM1's new 0.54 |
| **ASK 2** | 30 | **is this audio usable, yes or no** — the one that decides whether voice isolation becomes a detector |
| **ASK 3** | 6 | confirm FM4's files really are dead |
| **ASK 4** | 10 | does the crop look right |

**ASK 2 is the one that matters.** Everything else is a confirmation; that one
decides whether the 21 clips you found can be caught automatically or whether
their raster is the only handle we have on them.

---

## 7. What I did not do

- **Did not ship a voice-isolation detector.** It would reject about half of V00
  at any cut that catches your labelled clips. Measured and shown; not applied.
- **Did not re-fit any SMPL-H.** The anamorphic files are excluded rather than
  repaired; §1.4 of `reports/08_vendor_repairs.md` is the argument for when a
  re-fit would be worth it.
- **Did not change FM2 or FM3.** Neither has been contradicted by any review.
- **Did not re-render V00.** Every Round-7 and Round-8 repair is a no-op there —
  square pixels, upright, consistent timebase, no dead audio in 300 sampled
  files — so the only change would be the buffering script, which the rebuild
  picks up for free.
