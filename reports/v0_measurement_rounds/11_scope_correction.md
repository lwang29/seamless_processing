# Round 10: the seated clip that passed was a scope error, not a measure error

Three answers came back. Two closed their questions; the third opened a better
one.

| ask | answer | consequence |
|---|---|---|
| **ASK 2** — audio on surviving clips | all 30 usable | **Closed.** The raster exclusion handled the audio problem; no further detector. |
| **ASK 3** — the crop | bars gone, nothing lost at the edges | **Closed.** |
| **ASK 1** — posture, 32 more labels | 12 sitting, 20 standing | Hip flexion holds; the cut's cost is now measured properly. §2 |
| *(unprompted)* `V03_S1558_I00000012_P1425` passed FM1 while seated | **not a measure failure** | FM1 caught it and was overruled. §1 |

New gallery: `artifacts/private_review_vendors_r10/` — 175 clips, two asks.

---

## 1. The clip that passed: FM1 was right and got outvoted

The file's own reading is **92.6° of hip flexion** — as deeply seated as anything
in the label set — and `fm1_sitting_file` is `True`. It passed anyway because the
verdict is not taken from the file. It is taken from a majority of the
participant-session, and **13 of that session's 14 files read standing**, so the
session verdict is "standing" and the seated file inherits it.

That rule comes from your own Round-4 observation: a participant either stands in
all their clips or sits in all of them. Measurement in Round 4 put the scope at
the recording session rather than the person, and it has been right ever since —
**for V00**.

It is not right for V03:

| | multi-file participant-sessions | mixed (some files seated, some not) |
|---|---|---|
| V00 | 1,072 | **12 — 1.1%** |
| V03 | 1,293 | **210 — 16.2%** |

And in V03's mixed sessions the seated share has a **median of exactly 0.50**, so
the majority vote is close to a coin toss. On the 108 labelled V03 files present
in the scans:

| verdict scope | sitters caught | standing lost | accuracy |
|---|---|---|---|
| session majority | 35 / 46 | 13 / 62 | 0.778 |
| **the file itself** | **40 / 46** | **7 / 62** | **0.880** |

Worse on both axes at once. **V03 now takes FM1's verdict from its own file**;
V00, V01 and V02 keep the session rule. Both columns are always written, so
either can be recovered without a rescan.

The corpus effect is almost nil in volume — V03's FM1 goes 24.40% → 23.87% — and
substantial in who gets caught. 124 V03 files were seated-but-outvoted; 154 were
standing-but-outvoted.

Your other observation points the same way at a finer grain.
`V03_S1821_I00000010_P3624`, which you noted "starts standing, then sits down
about 5 seconds in, before standing up again a little after the 2 minute mark":
its hip flexion runs p25 = 98°, median 137°, p75 = 161°. Posture is not constant
within that *recording*, let alone within its session. The median still lands on
the right side of the cut, so the file is caught, but the spread is the honest
signature and it is now on the card.

---

## 2. Hip flexion, on 123 labels

Your 32 new answers take the V03 posture set to **123** (58 seated, 65 standing).
The measure holds:

| | hip flexion |
|---|---|
| 65 standing | min **136.1°**, p25 151.0°, median 159.9° |
| 58 seated | median 116.3°, p75 124.6°, max **151.9°** |

The cut at 146° now reads:

| cut | sitters caught | standing lost | accuracy |
|---|---|---|---|
| 138° | 54 / 58 | 3 / 65 | **0.943** |
| 144° | 55 / 58 | 5 / 65 | 0.935 |
| **146°** *(current)* | **57 / 58** | **8 / 65** | 0.927 |
| 152° | 58 / 58 | 21 / 65 | 0.829 |

So the honest number is 8 of 65 standing lost, not the 4 of 45 the smaller set
gave. That is still the trade you asked for — 98% of sitters — and 146° is still
far better than any knee-position cut, which at best managed 41 of 45 sitters for
12 of 45 standing. But the cost is larger than last round's estimate, and **ASK 2**
labels the 138–152 band so the choice between 138° and 146° rests on data rather
than on a preference stated once.

---

## 3. Where the vendors stand

Round-9 measurements, Round-10 verdict policy. Nothing about the media changed,
so the scan is reused rather than recomputed.

| vendor | files | FM0 | FM1 | FM2 | FM3 | FM4 | survives | dyad-hours |
|---|---|---|---|---|---|---|---|---|
| V01 | 1,494 | 46.05% | 0% | 68.41% | 8.57% | 3.21% | **15.93%** | 52 |
| V02 | 1,500 | 0% | 0% | 78.60% | 5.40% | 1.67% | **17.33%** | 132 |
| V03 | 1,500 | 3.07% | 23.87% | 80.27% | 7.00% | 0.53% | **10.80%** | 154 |

**338 dyad-hours** against V00's 419.

---

## 4. What I need from you

| | clips | question |
|---|---|---|
| **ASK 1** | 28 | sitting or standing, on the files the scope change moves — 14 each way |
| **ASK 2** | 20 | sitting or standing, hip flexion 138–152°, where the cut costs standing files |

Both are the same question and neither is open-ended. After these, FM1 on V03
has been fitted on roughly 170 hand labels and I would stop asking about posture.

## 5. What I did not do

- **Did not change the measure again.** Hip flexion holds on 123 labels.
- **Did not change V00's scope.** 98.9% of its sessions are unanimous and the
  session rule was validated there on 66 labels.
- **Did not look further at audio.** All 30 surviving clips were usable; the
  raster exclusion closed it.
- **Did not re-scan.** Round 10 reuses Round 9's measurements through a symlink,
  because only the verdict policy moved. Round 9's config now pins the old scope
  explicitly so it still reproduces its own numbers.
