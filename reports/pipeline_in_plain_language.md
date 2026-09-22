# The pipeline, in plain language

What the filter does, why each part is there, and what it is trying to catch.
No code, no formulas. The technical version is [`docs/pipeline.md`](../docs/pipeline.md).

---

## The one question

For every 30 seconds of every recording, the pipeline asks:

> **While this person is talking, are their hands and arms actually gesturing?**

Everything else serves that question. In particular, a clip can have perfect
video, perfect audio and perfect body tracking and still be thrown out, because
none of those things tell you whether the participant moved their hands. That
was the original complaint about the dataset, and it is the thing the pipeline
is built around.

The whole process is automatic. Nobody — no person and no model — looks at a
clip to decide whether it goes in.

---

## The shape of it

Think of it as a funnel with four narrowings:

1. **Which recordings are worth looking at at all?** (cheap, metadata only)
2. **Measure everything.** (expensive, done once)
3. **Which 30-second windows are usable?** (a wide net)
4. **Which usable windows are actually good co-speech gesture?** (the real decision)

We start with 129,370 recordings and end with 50,741 accepted clips — about
**385 hours**.

---

## Step 1 — Throw out recordings that can't be used

Before measuring anything, we drop recordings that are missing a file, have no
video, are too short to hold a single 30-second window, contain no speech at
all, or were filmed in one of four camera formats that the dataset's own body
tracking handles badly. We also drop the "charades" sessions, because those are
a game with scripted physical actions rather than a conversation.

**Why first:** measuring costs about a second per recording. There's no point
spending it on something unusable.

**Result:** 118,570 of 129,370 recordings survive.

*Caveat:* the camera-format rule is a blunt instrument. It removes formats known
to cause problems rather than detecting the problem itself, so some good
recordings in those formats are lost.

---

## Step 2 — Measure the motion, in a way that can't be fooled by the body moving

This is where two ideas do most of the work.

### Idea one: measure the arms relative to the torso, not the room

If someone sways in their chair, turns to face their partner, or the camera
drifts, their hands move a lot — in the room. But their arms haven't *done*
anything. So instead of measuring where the hands are in the room, we measure
where they are **relative to the person's own shoulders and spine**.

Swaying, leaning, turning and walking all become near-zero automatically. This
isn't a setting that can be tuned wrong; it's a change of viewpoint. It's the
main defence against counting whole-body movement as gesturing.

### Idea two: movement has to *go somewhere*, not just be fast

Body tracking is noisy. A hand that is perfectly still on video still jitters in
the tracking data, and that jitter is genuinely fast — several times faster than
the speed you'd need to call something "moving".

So speed alone is not enough. We require that the hand also **travels a real
distance** — at least 3.5 cm within half a second. A hand vibrating in place goes
nowhere, no matter how fast it vibrates, so it doesn't count.

### And then: counting separate gestures, not total motion

We group the moving frames into **episodes** — separate bursts of gesturing. This
turns "how much did they move" into "how many separate times did they gesture,
and for how long each time". That distinction is what separates someone
gesturing while they talk from someone who adjusted their glasses once.

Crucially, all of this is measured **only while the participant is speaking**,
using the speech timings that ship with the dataset.

---

## Step 3 — The wide net: is this window usable at all?

Eighteen checks. They cover:

- **Is the body tracking working?** Enough valid frames, no long unbroken
  stretch of failure, the person was actually detected, no physically impossible
  arm speeds.
- **Is there enough speech to judge?** At least 8 seconds of the participant
  talking.
- **Is there any gesturing during that speech?** Some active time, spread over
  at least three separate episodes, covering a reasonable share of what they
  said.
- **Do the arms visit more than one position?** This one was added after
  noticing that people with their hands clasped at their waist pass every
  speed-based test — their fingers shuffle fast enough — while never actually
  moving their hands anywhere.

These checks are deliberately **generous**. They run over 2.4 million windows and
their job is to remove what's *unusable*, not to judge how good it is. If they
were strict, they'd delete quiet gesturers along with non-gesturers.

**Result:** 17% of windows survive, which become **73,883 candidate clips**
(about 616 hours).

---

## Step 4 — The real decision

This is the step that replaced having a person look at each clip.

It works in two parts.

### Part one: nine disqualifiers

Each of these names something that is **not gesturing at all**, rather than
"less gesturing". Failing any one is fatal — nothing else can make up for it:

| the check | what it catches |
|---|---|
| enough of the speaking time is active | **hands basically still while talking** |
| the hands are carried high enough | hands parked in the lap or hanging at the sides |
| at least three separate episodes | a single movement |
| episodes last long enough | a string of twitches rather than gestures |
| gesturing spread across what they said | one adjustment, not sustained gesturing |
| arms moving more than the torso is | motion that's really whole-body movement |
| motion goes in consistent directions | tracking noise (see below) |
| two independent measurements agree | tracking noise (see below) |

**On the two noise checks.** We deliberately do *not* filter on "how big is the
jitter", because we measured that almost every way of doing that is really just
a measure of how much the person gestured — you'd delete the most expressive
people first. Instead:

- Real movement keeps going in a direction. Tracking glitches jump out and snap
  back. So we look at **direction**, not size.
- We measure each arm **two independent ways** — from the 3D body model and from
  the 2D video keypoints. Both can be noisy; they can't be noisy in the *same
  way* by accident. If they disagree, something is inventing motion.

### Part two: a score, not a checklist

For the rest — how much, how high, how sustained, how vigorous — we don't use a
list of strict thresholds. Stacking strict thresholds is how you quietly delete
everyone who's a bit reserved.

Instead we compute a **score out of four things**:

- **Posture** (40%) — how high the hands are carried, how much space they use,
  how far the elbows come from the body, whether the arms visit different places
- **Persistence** (30%) — how much of the speaking time is gesturing, and how
  long each gesture lasts
- **Vigour** (10%) — how fast the arms move
- **Integrity** (20%) — how clean and consistent the motion is

These are averaged with weights, so **being strong on several makes up for being
modest on one**. Someone who gestures small but keeps their hands up and works
them steadily through every sentence will pass. That's on purpose — it's the
safeguard against silently throwing away genuine but restrained gesturers, which
would be its own kind of failure.

Vigour is weighted lowest deliberately: how fast someone moves is mostly
personality, not evidence.

### One thing we deliberately ignore

**The single biggest arm movement in a clip is not counted as evidence.**

When we checked, clips that reviewers *rejected* had slightly *larger* peak arm
movements than clips they accepted. Rewarding big movements rewards exactly the
wrong thing: one dramatic isolated gesture.

The clearest example in the whole corpus is one participant who scores in the
top 10% on every activity measure. Ten of the twelve sampled moments show his
arms hanging at his sides. His entire score comes from twice adjusting his
beanie. If peak movement counted, he'd sail through.

**Result:** 69% of candidate clips qualify — **50,741 clips, about 385 hours**,
from 3,504 different participants.

---

## How do we know it works?

Two independent kinds of evidence.

**1. Built failure cases.** For each thing the pipeline is supposed to exclude,
we construct a synthetic recording that is exactly that failure and check it
gets rejected for the right reason:

| the constructed case | expected |
|---|---|
| Clean tracking, arms never move | rejected |
| Hand vibrating fast but going nowhere | rejected |
| Whole body sliding 40 cm, arms rigid | rejected |
| One big 0.8-second movement, then stillness | rejected |
| Large sustained gesturing, but only while silent | rejected |
| Clear two-handed gesturing while talking | **accepted** |
| One arm gesturing, the other in the lap | **accepted** |
| Modest gesturing, but steady and speech-locked | **accepted** |

That last row matters as much as the rejections. A filter that throws everything
away would pass all the exclusion tests.

**2. Comparison against real human judgements.** 100 recordings were reviewed by
hand, with audio, by someone applying the written criteria. Comparing the
automatic decision to those judgements:

- Of the clips the pipeline accepts, **93.6%** were also accepted by the human.
- Of the clips the human accepted, the pipeline keeps **91%**.
- It catches 8 of the 13 the human rejected.

For comparison: if you skipped this step entirely and took everything that got
through Step 3, 86% would be human-approved. So the step raises quality from 86%
to 93.6% while keeping 95% of the good material.

---

## What I'd want you to know about the limits

- **The thresholds are calibrated, not derived.** They come from agreeing with a
  labelled sample of real recordings. They're documented, they're all in one
  config block, and they're meant to be moved when there's evidence — not
  treated as physical constants.
- **It catches 8 of 13 rejects, not 13 of 13.** The remaining misses are clips
  that are genuinely marginal. Tightening further would start deleting good
  ones; that trade-off is set where it is on purpose and can be moved.
- **The validation set is small.** 100 hand-reviewed recordings, of which only
  13 were rejections. That's enough to show the step helps and roughly where it
  sits; it is not enough to state its accuracy to the decimal place.
- **One known blind spot.** The measure of "does the arm visit different places"
  uses an average. Someone who holds one position for most of the clip and a
  second position briefly can still score well on it — the beanie case above.
  Using a median instead would fix it, but that needs re-measuring the whole
  corpus.
- **Speech/gesture timing alignment is measured but not enforced.** We record how
  well gesture lines up with speech, but we don't reject on it, because we have
  no labelled data to set that threshold against. Saying otherwise would be
  overclaiming.
- **Lower body is not checked at all.** Out of scope by instruction, so nothing
  about the legs in these clips should be trusted.

---

## If you want to change how strict it is

Every threshold is in one block of `configs/vibes_upper_body.yaml` under
`qualify:`, each with a comment saying what it excludes. Re-running the decision
takes seconds and reads no video.

Every clip, kept or dropped, carries its own score, its four component scores,
and a list of **every** rule it failed. So "why was this clip dropped" and "what
would I get if I loosened this one rule" are both questions you can answer from
the output files, without re-running anything.
