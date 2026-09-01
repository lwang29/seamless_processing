# Round 9: one correction, two confirmations, and a bug I shipped

Three of your notes; three different kinds of answer.

| you said | answer |
|---|---|
| the black bars are still there | **A bug I shipped.** Round 8 implemented and tested the crop, then never called it. §1 |
| here are 42 more posture labels | They showed FM1 has been reading the **wrong measure** on V03. §2 |
| here are 30 audio labels | They **refuted** the voice-isolation idea. §3 |
| scrubbing still fails on a green banner | That rules out the transport, so this round stops using it. §4 |
| add a copy button | Done — the ⧉ next to each clip title. |

Net: **337 projected dyad-hours** across V01–V03, up from 313, because the better
FM1 measure catches more sitters while rejecting less of V03.

New gallery: `artifacts/private_review_vendors_r9/` — 213 clips, three asks.

---

## 1. The crop: implemented, tested, never called

You were right, and it is worth being precise about how it got through. Round 8
added `pillarbox` to the repair type, a measurement routine, an ffmpeg filter
equivalent, and four unit tests — and then built the renderer's repair object
without it:

```python
return RasterRepair(
    width=probe.width, height=probe.height,
    sample_aspect=probe.sample_aspect,
    quarter_turns=quarter_turns_from_roll(roll),
    # pillarbox never passed
)
```

Every test passed, because every test exercised the crop directly. Nothing
checked that the renderer *asked* for one. There is now a test that does exactly
that, and the fix is verified end to end rather than by unit test: 1080×960
renders at 540×960, 2180×3840 at 2160×3840, and the sidecar records
`cropped 540 black columns`.

Your read on 2180×3840 was slightly conservative, incidentally — it is 20 columns
on the right, not 10 on each side.

---

## 2. FM1 on V03 was reading the wrong measure

### What your labels showed

Your 42 answers took the V03 posture set to **90, evenly split 45/45**. On that
set, the measure FM1 has used since Round 4 does much worse than it looked on 58:

| | knee position along the torso |
|---|---|
| 45 standing | min 0.507, **p25 0.545**, median 0.572, max 0.724 |
| 45 seated | min 0.249, median 0.430, max 0.630 |

The standing quartile is at 0.545, so the 0.54 cut I set last round sits *inside*
the standing distribution. On the full 90:

| knee cut | sitters caught | standing lost |
|---|---|---|
| 0.50 | 36 / 45 | 2 / 45 |
| **0.54** *(Round 8)* | 41 / 45 | **12 / 45** |
| 0.56 | 43 / 45 | 20 / 45 |

Twelve of 45, not the four of 22 the smaller set implied. My Round-8 choice was
fitted on a sample that flattered it.

### The fix is a different measure, not a different cut

So I tested every signal in the scan against the 90 labels. Hip flexion wins, and
not narrowly:

| signal | AUC on all 90 | AUC inside the ambiguous band |
|---|---|---|
| **hip flexion** | **0.996** | **0.977** |
| knee flexion | 0.966 | 0.974 |
| knee position along the torso | 0.945 | — *(the band is defined by it)* |
| knee spread over torso | 0.875 | 0.895 |

As a rule, against the Round-8 one:

| rule | sitters caught | standing lost | V03 files rejected |
|---|---|---|---|
| knee position < 0.54 | 41 / 45 | 12 / 45 | 33.5% |
| **hip flexion < 146°** | **44 / 45** | **4 / 45** | **26.3%** |

Better on all three axes at once. Leave-one-session-out gives 43/45 and 3/45,
accuracy 0.944 against 0.822. And the two classes are genuinely separated on this
measure — standing bottoms out at **136.1°** and seated tops out at **151.9°** —
so 146° sits inside a real gap rather than on a knife edge. Adding knee position
back as an OR clause triples the standing loss and catches nothing extra.

### The honest caveat, and why I still believe it

Your labels were drawn from bands of *knee position*, so that measure's apparent
performance is depressed by the selection and hip flexion's is not. That is a
real confound. Three things survive it:

1. **Inside the ambiguous band** — knee position 0.50 to 0.63, where 9 seated and
   38 standing files overlap — hip flexion still separates at AUC 0.977. Since
   the bands were drawn on knee position, this comparison is conditional on the
   selection rather than biased by it.
2. Selection bias cannot manufacture a rule that catches **more** sitters while
   rejecting **less** of the vendor. Those pull in opposite directions.
3. The 136°/152° gap is a property of the two distributions, not of the sampling.

**ASK 1** brackets that gap so a wrong answer would show up immediately.

### An irony worth recording

Hip flexion was **retired in Round 5** because on 66 V00 labels every file it
flagged that knee position did not was a standing person. That was correct for
V00 and it is still correct for V00. It is simply the wrong conclusion to have
generalised, and V03's seating is different enough to reverse it — which is the
same lesson as Round 6's raster finding, arriving from another direction. FM1 is
now per-vendor in **which measure it reads**, not only where the cut sits.

---

## 3. Voice isolation: refuted

Round 8 proposed that the glitchy audio is a far-field or shared microphone,
measurable as poor separation between a speaker's own voice and their partner's.
The mechanism holds up. **The measure does not predict your verdict.**

Your 30 labels:

| | voice isolation (dB) |
|---|---|
| 27 usable | −1.68, 0.09, 0.34, 0.48, 0.93, 1.36, … up to 11.65 |
| 3 unusable | 0.33, 2.07, 2.17 |

The sets interleave, and there is an accepted clip at 0.34 dB against a rejected
one at 0.33. Any cut catching all three rejects also loses nine of your 27
accepts. There is no threshold here, and I am not going to invent one.

So voice isolation stays a **recorded signal** and no detector was built on it.
The card labels it as such.

**What actually handles the problem is your own raster exclusion.** All three
clips you rejected this round are 3840×2160, and 20 of the 21 you reported in
Round 7 are in rasters FM0 now drops outright. Exactly one — a portrait V03
file, `V03_S1676_I00000008_P4821` — is in a raster we keep, and nothing catches
it. **ASK 2** draws from survivors only, with no reference to the isolation
number, and asks the question that is left: is there bad audio still getting
through?

### FM4 stands

All six clips you checked were confirmed dead, and FM4 fired on all six. Nothing
to change.

---

## 4. Scrubbing: the transport is ruled out, so stop using it

A green banner means the server answered a range request with `206`. A green pill
means the clip was wholly inside `video.buffered`. Both true and seeking still
failing rules out the network and the file, and points at the media element's
**resource loader** — which is exactly what an embedded webview replaces with its
own, and those commonly refuse seeks whatever the server does.

I also re-checked the container, since it was the remaining suspect: moov first,
30 sync samples in a 30-second clip, `mvhd` and `mdhd` durations agreeing at
29.997 s, a standard one-entry edit list. Nothing a browser should struggle with.

So this round removes the loader from the path. Each clip is **fetched into
memory and handed to the element as a blob URL** — no ranges, no streaming, no
resource loader, just bytes the page already owns. The card still streams its
first frame immediately so the page reads as a contact sheet, and swaps to the
blob underneath; the pill turns green when it has:

> `streaming` → `loading into memory…` → `in memory · scrubbing works`

Only clips near the viewport are fetched and the oldest are released past a cap
of fourteen, so a 213-clip page holds about 60 MB rather than 900.

If it *still* fails after that, the problem is downstream of anything the page
can control, and the next thing to try is a different browser — but there is very
little left between a blob URL and the decoder.

---

## 5. Where the vendors stand

Round-9 sample, 4,494 files, fresh seed disjoint from Rounds 6–8:

| vendor | files | FM0 | FM1 | FM2 | FM3 | FM4 | survives | dyad-hours |
|---|---|---|---|---|---|---|---|---|
| V01 | 1,494 | 46.05% | 0% | 68.41% | 8.57% | 3.21% | **15.93%** | 52 |
| V02 | 1,500 | 0% | 0% | 78.60% | 5.40% | 1.67% | **17.33%** | 132 |
| V03 | 1,500 | 3.07% | 24.40% | 80.27% | 7.00% | 0.53% | **10.73%** | 153 |

**337 dyad-hours** against V00's 419, up from 313 last round: the better FM1
measure recovers 23 hours on V03 *and* catches three more sitters in every
hundred. V03's FM1 now fires 293 times on hip flexion, 58 on the shin add-on and
7 on both.

FM2 remains the dominant cost everywhere and is unchanged since Round 5.

---

## 6. What I need from you

| | clips | question |
|---|---|---|
| **ASK 1** | 32 | sitting or standing, bracketing the new 136–152° gap |
| **ASK 2** | 30 | **is the audio usable** — drawn only from clips that survive |
| **ASK 3** | 10 | are the black bars actually gone |

ASK 2 is the one that decides something. If the surviving clips all sound fine,
the raster exclusion closed the audio problem and I will stop looking for a
detector. If bad audio still gets through, those clips are the evidence for what
to measure instead — and I already know at least one such file exists.

---

## 7. What I did not do

- **Did not build a voice-isolation detector.** Refuted by your labels; §3.
- **Did not re-tune V00.** Hip flexion was tested there in Round 5 on 66 labels
  and lost. V00 keeps knee position at 0.43.
- **Did not change FM0, FM2, FM3 or FM4.** None has been contradicted.
- **Did not re-render the earlier galleries** with the crop fix or the blob
  player. Rounds 7 and 8 are answered; re-rendering them would cost an hour of
  compute to change artifacts nobody is reading.
