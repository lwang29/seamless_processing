# Reports

`pi_update_2026-09-30.md` is the current progress update for the PI. Every other
report here is a **historical record of the previous (filtering) iteration** and
its v0 measurement rounds. None of those describes the current pipeline, which is
documented in `docs/pipeline.md`; each carries a banner saying so.

| report | what it is now |
|---|---|
| `17_cospeech_gesture.md` | design of the arm-motion measures that are reused unchanged (torso frame, speed and travel, episodes) and the filter built on them |
| `18_automated_qualification.md` | calibration of the removed tier-2 decision against 90 human verdicts; which measures separated accept from reject |
| `pi_update_2026-09-30.md` | current: progress update on the annotation build, the judgment calls in it, and open questions for the PI |
| `v0_measurement_rounds/` | rounds 1-15: the evidence behind the posture thresholds (FM1: 164 V03 and 66 V00 labels), the dead-audio rule (FM4), camera geometry, vendor rasters and repairs |

The current run's numbers are in `<outputs.root>/annotations/annotation_report.md`.
