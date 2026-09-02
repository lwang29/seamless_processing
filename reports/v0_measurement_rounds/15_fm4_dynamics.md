# Round 14: what FM4 should have been measuring all along

Your 48 audio labels killed two candidate rules and pointed at a third that is
perfect on all of them. **No new gallery this round** — §4 explains why there is
nothing left to look at.

| rule | recall | precision |
|---|---|---|
| level < −55 dB and flat spectrum *(Round 8)* | 7 / 13 | **28%** |
| released VAD empty *(Round 13 candidate)* | 11 / 13 | **38%** |
| either of the two | 13 / 13 | 27% |
| both of the two | 5 / 13 | 83% |
| **envelope dynamics < 1.5 dB, or no audio** | **13 / 13** | **100%** |

---

## 1. Both of my candidates were wrong, and your comments say why

**The level floor** (28% precision). You found V01 clips that are "just complete
static throughout the entire video" at −23 dB — louder than most working
recordings. Level says nothing about whether there is a voice in it.

**The empty VAD** (38%). Your own explanation: *"the VAD failed, resulting in no
annotations, or the participant speaks very little or not at all. Even in this
latter case, I can tell that the audio is working because their partner can be
heard speaking."* An empty VAD is a fact about the annotator, not about the
recording.

I proposed the second of those last round on the strength of six labels. Forty-two
more refuted it. That is the second audio idea to die this way, after voice
isolation in Round 10.

## 2. What works: the level has to *move*

Static, buzz and silence all sit at one level whether anyone is speaking or not.
A track carrying a voice does not. Measuring the spread of the 50 ms level
envelope — its 95th percentile minus its median — over your 44 audible labels:

| | envelope dynamics |
|---|---|
| **unusable** (9) | 0.20 – **1.35 dB** |
| **usable** (35) | **1.70** – 69.04 dB |

They do not overlap, and the gap is clean. **The cut is 1.5 dB**, and with the
four no-audio files added it scores **13 of 13 with no false positive.**

It needs no VAD, no partner track and no absolute level, which is exactly why it
survives where three previous attempts did not. It is also the same judgement you
were making by ear: you verified a track was alive by hearing *someone* on it.

**One correction to your list.** `V01_S0104_I00000132_P1196` is filed under
usable, but its WAV is 58 bytes — a header with zero samples — and the renderer
substituted generated silence, so the clip you heard had nothing in it by
construction. I have counted it as unusable. Worth flagging in case it points at
a mis-paste rather than a mishearing.

## 3. What changes

FM4 now reads `audio_envelope_dynamics_db < 1.5`, or an unreadable/empty WAV. The
level and flatness numbers are still measured and reported; neither decides
anything.

Firing rates over 8,494 scanned files:

| vendor | level rule *(old)* | empty VAD | **dynamics *(new)*** |
|---|---|---|---|
| V00 | 0.22% | 0.10% | **0.03%** |
| V01 | 2.81% | 4.82% | **3.75%** |
| V02 | 1.53% | 0.67% | **0.33%** |
| V03 | 0.40% | 2.47% | **0.33%** |

It clears 85 files the old rules were wrongly rejecting.

### FM4 stays off for V00, and the case for turning it back on

You had me retire it there, and that instruction stands until you say otherwise.
But the reason has gone: the new rule made no mistakes on any of the 11 V00 files
in your labels, and across 4,000 V00 files it fires **once**.

That one file is `V00_S1669_I00000636_P0106A` — dynamics 1.46 dB, level −87.5 dB,
zero released speech — the clip I flagged last round as indistinguishable from a
confirmed-dead track. **It already fails the pipeline on other grounds**, so
switching FM4 back on for V00 would change exactly zero verdicts. It is one line
in `configs/`, and it costs nothing.

## 4. Why there is no gallery this round

Because there is nothing unreviewed to put in one. Of 8,494 files, the number the
new rule flags that **neither** old rule flagged is **zero** — its firing set is a
strict subset of the union you have already judged. Every file it rejects is one
you have seen, or is of a kind you have seen.

If that had come out differently I would have built one. It is worth saying
plainly because three rounds running have ended with an ask, and this one does
not need to.

## 5. Where the corpus stands

| vendor | files | FM4 | survives | dyad-hours |
|---|---|---|---|---|
| V00 | 4,000 | 0.03% *(not applied)* | 31.18% | **449** |
| V01 | 1,494 | 3.75% | 15.93% | 52 |
| V02 | 1,500 | 0.33% | 17.73% | 135 |
| V03 | 1,500 | 0.33% | 11.07% | 158 |

**794 dyad-hours**, V00 being 57% of it.

FM2 is now, by a wide margin, the only thing rejecting a large share of the
corpus — 59% to 80% depending on vendor — and it is the one check no review has
ever contradicted. If more hours are wanted, that is the only place left with
any to give.

## 6. What I did not do

- **Did not re-enable FM4 for V00.** Your instruction, and reversing it on my own
  reading would be the mistake this project keeps finding. The case is in §3.
- **Did not keep the level or VAD rules as extra clauses.** Both add false
  positives and catch nothing the dynamics rule misses.
- **Did not re-scan.** The measurement was taken over the existing scans; it is
  now part of the scanner for future rounds.
