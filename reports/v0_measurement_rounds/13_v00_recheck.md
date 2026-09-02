# Round 12: V00 re-examined under the four lessons V03 taught

Three of the four V03 corrections were to choices that had looked settled on V00
evidence. V00 is 1,441 dyad-hours, the largest block in the corpus, and its
settings had not been revisited since Round 5. So all four were re-tested here.

**All four V00 settings survive.** What does not survive is the *coverage* of the
labels they rest on — and that gap is what the new gallery is for.

`artifacts/private_review_v00_r12/` — 142 clips, four asks.

---

## 1. The four lessons, applied to V00

### Lesson 1 — raster. Closed by census, not by sample.

Over **all 41,205 eligible V00 files**, not a draw:

| | |
|---|---|
| rasters | 1080×1920 — **one**, no exceptions |
| frame rates | 30/1 — one |
| empty WAVs | **0** |
| files whose annotations miss their video by >0.5 s | **0** |

And over the fresh 4,000-file scan: one distinct pixel aspect (1.0), zero files
needing a quarter turn, zero with a body roll beyond 10°. **FM0 fires on nothing
in V00.** Every defect that cost V01 44% of its files and V03 3% is simply absent
here.

### Lesson 2 — the measure. Knee position is right for V00.

V03 switched from knee position to hip flexion because hip flexion was far
better there. Round 5 had tested hip flexion on V00 only as an *extra clause*,
never as a **replacement**, so that was the test to run. On V00's 66 labels:

| measure | AUC | best operating point |
|---|---|---|
| knee position | 0.9283 | **17 / 19 sitters for 1 / 47 standing lost** (cut 0.43) |
| hip flexion | 0.9317 | 17 / 19 sitters for 3 / 47 standing lost (cut 115°) |

Almost identical separating power, and knee position converts it better. **No
change.**

Why the two vendors differ is visible in one number — standing participants'
median hip flexion:

| | V00 | V03 |
|---|---|---|
| standing | **129.0°** | **159.3°** |

A 30° offset. V00's standing participants read as considerably more hip-flexed
than V03's, which is why no shared hip cut could ever have worked, and why hip
flexion has much less headroom above the seated range here.

### Lesson 3 — verdict scope. Session majority is right for V00.

| scope | sitters caught | standing lost | accuracy |
|---|---|---|---|
| **session majority** | 18 / 19 | **0 / 47** | **0.985** |
| per-file | 18 / 19 | 1 / 47 | 0.970 |

The opposite of V03, and for the reason the rule assumes: **1.1%** of V00's
multi-file participant-sessions are mixed, against V03's 16.2%. The assumption
holds here. **No change.**

### Lesson 4 — the shin add-on. Keep it.

On V03 it flagged 8 standing people and no sitters. On V00 it fires on
**1 file in 4,000** — `V00_S0216_I00000515_P0293` — and that file is one you
confirmed seated. It adds one true positive and zero false ones:

| rule | sitters | standing lost |
|---|---|---|
| knee < 0.43 alone | 17 / 19 | 1 / 47 |
| knee < 0.43 or shin inverted | **18 / 19** | 1 / 47 |

**No change.**

---

## 2. What the re-test could not settle: where V00's labels come from

All 66 V00 posture labels were drawn from the **flagged pool** or the **Round-4
adjudicate pool** — files FM1 was uncertain about. They therefore cluster around
the 0.43 cut by construction:

| | knee position |
|---|---|
| 47 standing | min 0.394, p25 0.482, median 0.517, max 0.613 |
| 19 seated | p25 0.299, median 0.357, p75 0.387, **max 0.728** |

Two things follow. First, that single seated file at 0.728 shows V00 *does* have
sitters far above the cut — it is only caught at all because the shin add-on
happens to fire on it. Second, and more important:

**FM1's false-negative rate over the comfortably-passing V00 population has never
been measured.** Of 4,000 sampled files, 3,485 pass FM1 with a knee position
above 0.50, and not one file in that region has ever been looked at by eye.

That is precisely where the V03 problem lived: seated participants at 0.435–0.586,
well clear of a 0.43 cut, found only because you happened to see one. The V00
equivalent would be invisible to every measurement I can make, because the
measurement is the thing under test.

**ASK 1** is a ladder across the whole pass side, 0.43 to 0.80, to settle it.

---

## 3. New checks V00 had never had

| check | result |
|---|---|
| FM0 (raster, timebase) | fires on **0** of 4,000 |
| FM4 (dead audio) | fires on **9** of 4,000 — 0.22%, against V01's 3.21% |
| voice isolation | median **7.65 dB**, in the close-mic band, as expected |

The nine FM4 files are the first dead audio found in V00; the earlier 300-file
probe found none, which at 0.22% is what you would expect from 300 draws.
**ASK 4** shows them.

---

## 4. Where V00 stands

Fresh 4,000-file draw, seed 20270210, disjoint from the tuning and briefing
samples:

| condition | files | FM0 | FM1 | FM2 | FM3 | FM4 | survives | dyad-hours |
|---|---|---|---|---|---|---|---|---|
| improvised | 2,000 | 0% | 1.20% | 66.20% | 3.70% | 0.05% | **30.70%** | 298 |
| naturalistic | 2,000 | 0% | 3.10% | 58.95% | 10.55% | 0.40% | **31.65%** | 149 |

**447 dyad-hours**, against 341 across V01–V03 combined. V00 remains the bulk of
the usable corpus, and the two new checks cost it almost nothing.

Naturalistic trips FM3 nearly three times as often as improvised (10.55% against
3.70%) — participants gesture less when not performing a task. That was visible
in Round 2 and is unchanged.

---

## 5. What I need from you

| | clips | question |
|---|---|---|
| **ASK 1** | 40 | sitting or standing, across the whole FM1 **pass** side — the false-negative hunt |
| **ASK 2** | 16 | sitting or standing, 0.30–0.43, the reject side |
| **ASK 3** | 11 | sitting or standing, files a session outvoted, both directions |
| **ASK 4** | 6 | confirm FM4's nine V00 files really are dead |

**ASK 1 is the one that matters.** If those 40 are all standing, FM1 on V00 is
confirmed on a population its labels have never covered and I would consider
posture closed for the whole corpus. If sitters turn up above 0.50, V00 has V03's
problem and its cut needs re-fitting on labels drawn the right way.

---

## 6. What I did not do

- **Did not change any V00 threshold.** All four survived their re-test; changing
  one on the strength of a V03 result would be the mistake this round exists to
  avoid.
- **Did not re-scan V01–V03.** Nothing here affects them.
- **Did not test voice isolation again.** Refuted in Round 10 and V00's median
  sits in the close-mic band anyway.
