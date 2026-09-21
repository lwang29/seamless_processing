# Review rubric — co-speech upper-body gesture

This is the exact text every reviewer works from, human or model. It is kept in
the repository rather than in someone's head so that verdicts taken months apart,
or by different people, mean the same thing — and so that a disagreement can be
traced to a rule rather than to a mood.

**The unit is a review item = one participant file.** Accepting it accepts that
file's *accepted spans*: the contiguous 30-second windows the automated gates
selected, marked in green on the card's timeline. The card's twelve thumbnails
are sampled from inside those spans, so what you are looking at is what would
enter the training set — not a random moment of the recording.

**What the subset is for.** An initial ViBES upper-body training experiment.
Only the torso, shoulders, arms, wrists, hands, neck and head matter.

---

## Accept when all four hold

1. **The upper-body SMPL-H tracking is valid.** In the photo rows, the drawn
   arms and hand points sit on the participant's actual arms and hands. In the
   pose rows, the skeleton is a plausible human upper body.
2. **The arms and hands genuinely move.** Across the twelve moments the arms and
   hands take visibly different positions. Not the same posture twelve times
   with the fingers slightly re-arranged. **One arm is enough** — see below.
3. **The motion is natural, not noise.** Poses look like a person gesturing:
   smooth, anatomically sensible, no popping between unrelated poses, no
   impossible joint bends, no limb detached from the body.
4. **Gesture and speech go together.** On the timeline, green gesture episodes
   coincide with blue own-speech bars inside the green accepted spans. If you
   play the clip, the hands should move with the person's talking rather than
   independently of it.

## Reject, with the reason that fits best

| reason | use it when |
|---|---|
| `static_hands` | the hands barely move while the person is speaking; hands parked in a lap, clasped at the waist, in pockets, or hanging at the sides throughout |
| `not_co_speech` | there is movement, but it is unrelated to their speaking — sustained fidgeting, adjusting clothing or hair, handling an object, or motion that only happens while the *partner* talks |
| `tracking_broken` | the SMPL-H arms are not on the participant's arms; the skeleton drifts off, collapses, or tracks the wrong person |
| `unnatural_motion` | pose pops between unrelated configurations, joints bend the wrong way, hands flicker, or the motion is visibly the tracker rather than the person |
| `out_of_sync` | gesture and speech are clearly not aligned — long gesture bursts in silence and stillness during speech |
| `obscured` | the hands are not visible enough to judge, or would not be learnable: out of frame for most of the spans, behind a desk, or holding something that hides them |
| `other` | anything else; always add a note |

## Explicitly **not** reasons to reject

These come straight from the PI's brief and from what the v0 pipeline got wrong.

- **Seated posture.** A person sitting on a chair or stool is fine.
- **Legs out of frame, or lower-body tracking that is obviously wrong.** Only the
  upper body is being trained. The card does not even show the legs.
- **A hand briefly leaving the image**, as long as the upper-body and hand pose
  stays stable and natural while it does.
- **Camera framing, tilt, or a plain background.** Those were the previous
  round's questions and are not this one's.
- **Being a quiet participant.** How much of the *whole recording* they speak in
  does not matter; the spans were chosen for speech density already.
- **Small hand and finger motion on its own** is not automatically a reject —
  but it is only an accept if the arms move too. Beat gestures at chest height
  count; twiddling clasped fingers at the waist does not.
- **One-handed gesturing.** A participant who gestures actively and in time with
  their speech using one arm, while the other rests in their lap or on a chair
  arm, is an **accept**. Judge the gesturing arm on its own merits: if that arm
  alone satisfies the four criteria above, accept. Do not mark these `unsure`,
  and do not reject them as `static_hands` because the other hand is still.

  *Decided 2026-09-21, by Logan Wang, pending the PI's confirmation.* This is a
  rubric change only — no gate moved. Every activity and posture measure in
  `gates.py` was already computed as the **maximum over the two hands** (the
  more mobile wrist, the higher wrist, the more abducted arm), so the automated
  filter has always passed one-handed gesturing; only the written rubric was
  silent, and reviewers were resolving that silence as `unsure`.

## Use `unsure` freely

`unsure` marks the item as seen and keeps it out of the accepted manifest. It
costs nothing: the candidate pool is many times larger than the review budget,
so a hard case is better skipped than argued over. Use it when the card is
genuinely ambiguous and playing the clip does not settle it.

## When to play the clip

The card answers most items. Play the 30-second clip (`V` in the review app)
when:

- gesture and speech look plausibly related but you want to hear it;
- the motion might be tracker noise rather than the person;
- you are about to reject an item that the metrics say is strong, or accept one
  they say is weak.

Verdicts taken with the clip playing are recorded with `saw_video: true` and are
the ones that appear in `accepted_clips_with_audio.csv`.

## Reading the card

```
 header      item id, file id, vendor/condition, recording length, clips and seconds accepted
             then the automated measures for the accepted spans
 rows 1, 3   video, cropped to the upper body, with the released 2D arms (yellow)
             and hand points (orange = left, blue = right) drawn on
 rows 2, 4   the same moments as pelvis-frame SMPL-H, three-quarter view: this is
             the pose ViBES would train on, with global body motion removed
 timeline    the whole recording: own speech (blue), partner speech (brown),
             arm speed (white trace), gesture episodes (green shading),
             accepted spans (green bars), thumbnail times (ticks)
```

A pose panel labelled `x0.NN` was shrunk to fit an unusually large gesture; a
panel outlined in red is a frame the release flagged `smplh:is_valid == False`.
