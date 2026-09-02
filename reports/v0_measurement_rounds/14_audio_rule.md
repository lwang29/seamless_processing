# Round 13: FM1 on V00 is confirmed, and FM4 is measuring the wrong thing

Your three V00 answers closed three questions and opened one better one.

| ask | answer | consequence |
|---|---|---|
| **ASK 1** — 40 clips from the FM1 pass side | **all standing** | FM1 on V00 confirmed where its labels had never looked. §1 |
| **ASK 2** — 16 clips below the cut | 9 sitting, 7 standing | The cut costs ~9 dyad-hours. Keep it. §1 |
| **ASK 3** — 11 outvoted files | 8 standing, 3 sitting | Session-majority scope right on 8 of 11. Keep it. §2 |
| **ASK 4** — 6 FM4 firings | **none dead** | FM4 off for V00, as you asked — and the reason generalises. §3 |

New gallery: `artifacts/private_review_audio_r13/` — 157 clips, three asks, all
about the audio rule.

---

## 1. FM1 on V00 is confirmed

ASK 1 was the one that mattered: a random ladder across the whole FM1 **pass**
side, the population its 66 labels had never covered, and the place the V03
problem was hiding.

**0 seated in 40**, evenly across four bands from 0.43 to 0.80:

| knee position | clips | seated |
|---|---|---|
| 0.43–0.50 | 10 | 0 |
| 0.50–0.57 | 10 | 0 |
| 0.57–0.64 | 10 | 0 |
| 0.64–0.80 | 10 | 0 |

By the rule of three that puts a 95% upper bound of **7.5%** on the pass-side
seated rate, with a point estimate of zero. **V00 does not have V03's problem.**

ASK 2 measured the other side. In 0.30–0.43, **9 of 16 are seated** — so FM1's
precision there is 56%, and the 7 standing files it discards are real cost. That
band is 1.45% of V00, so the loss is about **0.63% of the vendor, 9 dyad-hours of
1,441**. Small enough to leave alone, and moving the cut down would trade those 9
hours for seated files re-entering the corpus, which is the wrong direction.

**One caution on the older numbers.** The 66 pre-existing V00 labels came from
the flagged and adjudicate pools, so they cannot give a population rate — mixing
them with these two random draws would have implied a 3.5% seated rate on the
pass side, which ASK 1 refutes outright at 0/40. Only the two fresh draws are
used above.

---

## 2. The session-majority scope is right for V00

The exact opposite of V03, and for the reason the rule assumes.

| direction | n | file rule right | session rule right |
|---|---|---|---|
| file says seated, session said standing | 7 | 1 / 7 | **6 / 7** |
| file says standing, session said seated | 4 | 2 / 4 | 2 / 4 |
| **both** | 11 | 3 / 11 | **8 / 11** |

The six the session rule correctly overrode all sit at knee 0.403–0.430 — right
at the cut, where the per-file measure is noisiest. The rule is doing exactly
what it was introduced to do. The one genuine seated file it lets through is
`V00_S0216_I00000515_P0293`, at 0.728, which the shin add-on catches anyway.

**No change.** V03 needed per-file because 16.2% of its sessions are mixed; V00
is 1.1%.

---

## 3. FM4: right instruction, and the reason generalises

You asked me to remove FM4 for V00. Done. But the six clips explain *why* it
failed, and the explanation implicates the rule everywhere.

### The floor is absolute; the vendors are not

| vendor | median speech level | where FM4's −55 dB floor sits |
|---|---|---|
| V01 | −18.6 dB | 36.4 dB below the median |
| V03 | −19.9 dB | 35.1 dB below |
| V02 | −37.1 dB | 17.9 dB below |
| **V00** | **−41.8 dB** | **13.2 dB below** |

V00's 1st percentile is −56.9 dB, so the floor cuts into its normal population.
On V01 nothing but a dead track ever gets near it. Same rule, entirely different
meaning.

### Making the floor relative does not fix it

The obvious generalisation — measure the floor against each vendor's own median —
was tested and only halves the problem:

| rule | catches the 6 confirmed dead | fires on your 6 V00 clips |
|---|---|---|
| absolute, < −55 dB | 6 / 6 | **6 / 6** |
| 30 dB below the vendor median | 6 / 6 | 3 / 6 |
| 25 dB below the vendor median | 6 / 6 | 4 / 6 |

It cannot work, because **"speaks rarely" and "no voice at all" look identical to
any whole-file average.** A participant who says forty words in five minutes has
the same mean level as one who says nothing.

### What does separate them: seconds of released speech

| | seconds of released VAD | whole-file level |
|---|---|---|
| your 6 V00 clips | 1.6, 2.7, 6.9, 28.5, 79.1, **0.0** | −56.9 to −85.7 dB |
| 6 confirmed dead | **0.0** in every case | −57.5 to −90.2 dB |

And across 400 random files per vendor, an empty VAD occurs in:

| V00 | V01 | V02 | V03 |
|---|---|---|---|
| **0.00%** | 5.25% | 0.75% | 3.25% |

Zero in V00 — exactly the behaviour you asked for, arrived at by measurement
rather than by exception.

**One clip does not fit.** `V00_S1669_I00000636_P0106A` has zero VAD *and* a
−85.7 dB whole-file level, indistinguishable by any measurement I have from the
three V03 files you confirmed dead — yet you judged it not dead. It is in the
gallery.

### Why I have not swapped the rule in

Because outside V00 the two rules disagree about 84 files, and that difference is
unvalidated:

| | VAD empty, level floor silent | level floor fires, speech present |
|---|---|---|
| V00 | 4 | 0 *(FM4 retired)* |
| V01 | 26 | 2 |
| V02 | 3 | 18 |
| V03 | 30 | 1 |

Swapping would newly reject 63 files and newly keep 21, and I am not going to
make that trade on the strength of six labels. **The gallery is those 84 files,
in both directions, plus a control where the rules agree.** `audio_vad_seconds`
is now measured and recorded either way.

---

## 4. Where the corpus stands

V00 with FM4 retired; V01–V03 unchanged from Round 11.

| vendor | files | FM0 | FM1 | FM2 | FM3 | FM4 | survives | dyad-hours |
|---|---|---|---|---|---|---|---|---|
| V00 | 4,000 | 0% | 2.15% | 62.58% | 7.12% | — *(off)* | **31.18%** | 449 |
| V01 | 1,494 | 46.05% | 0% | 68.41% | 8.57% | 3.21% | 15.93% | 52 |
| V02 | 1,500 | 0% | 0% | 78.60% | 5.40% | 1.67% | 17.33% | 132 |
| V03 | 1,500 | 3.07% | 20.00% | 80.27% | 7.00% | 0.53% | 10.80% | 157 |

**790 dyad-hours** across the corpus, of which V00 is 57%.

---

## 5. What I need from you

| | clips | question |
|---|---|---|
| **ASK 1** | 24 | is the audio usable — files the VAD calls empty and the level floor does not |
| **ASK 2** | 19 | is the audio usable — files the level floor rejects that do contain speech |
| **ASK 3** | 6 | the control, where both rules agree |

All three are the same yes/no. Three of ASK 1 are V00, and those matter most: a
rule that brings V00 back inside FM4 has to be right about them.

---

## 6. What I did not do

- **Did not change any V00 posture threshold.** All four settings survived
  Round 12's re-test and ASK 1 confirmed the one thing that was untested.
- **Did not swap FM4's rule.** 84 unvalidated disagreements; that is what the
  gallery is for.
- **Did not remove FM4 anywhere but V00.** The instruction was V00-specific and
  the evidence is V00-specific.
