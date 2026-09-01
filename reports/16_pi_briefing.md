# Round 15: the corpus-wide briefing gallery

A single page covering all four vendors under the pipeline as it now stands, for
your PI to read cold and check against the footage himself.

`artifacts/private_briefing_all_vendors/index.html` — **246 clips, 22 sections,
227 participants**, on a fresh random sample nothing has been tuned against.

```bash
$VIBES scripts/serve_gallery.py artifacts/private_briefing_all_vendors --port 8000
ssh -N -L 8000:localhost:8000 <cluster-host>     # then http://localhost:8000/
```

---

## 1. What it shows

| vendor | files scanned | kept | projected dyad-hours |
|---|---|---|---|
| V00 | 1,500 | **30.3%** | 436 |
| V01 | 1,496 | 16.6% | 55 |
| V02 | 1,500 | 18.1% | 138 |
| V03 | 1,500 | 10.7% | 153 |
| **all four** | **5,996** | **19.7%** | **781 of 3,961 eligible** |

One fresh draw of 1,500 per vendor, seed 20270420, disjoint from every sample any
threshold was fitted on. The scan produced 5,996 usable rows and 4 files with
zero-length annotation arrays.

## 2. Everything you asked for

| | |
|---|---|
| descriptions of each check | five cards, each with what it is for and how it decides |
| table of contents | **two levels** — V00–V03 as top-level entries, the seven strata nested inside |
| filtering metadata | headline stats, a per-vendor rate table, and projected dyad-hours |
| clip-level information | ten rows per card, each labelled in words with its cut point stated |
| scrubbing | the blob-loading fix is intact; the readiness pill is **gone** |
| copy button | ⧉ beside every clip title |
| download button | the full recording, beside every clip |
| stratification | per vendor: improvised-kept, naturalistic-kept, and one section per check |
| absent strata | named and explained in their own panel, §4 |

**"The three checks" is now "The five checks."**

**The table.** Its numeric column headers were left-aligned while the numbers
under them were right-aligned, so the labels floated left of their digits. Header
and body are both nine cells; the fix was to right-align the headers of numeric
columns, add a totals rule, and stop the columns wrapping.

**Clip signals.** Each row now names the check, says what it measures in plain
words, and states the cut inline — *"FM3 · share of the recording with both hands
parked (rejects above 75%)"* — so a number can be read without knowing the
codebase. Raw column names appear nowhere.

## 3. One thing the table does that is easy to miss

A dash means **the check is not applied to that vendor at all**, which is not the
same as applying it and finding nothing. V00's FM0 cell reads `0.0%` because FM0
runs on V00 and finds nothing; V00's FM4 cell reads `—` because FM4 is switched
off there. The bottom row averages each check only over the vendors it is applied
to, so a switched-off vendor cannot dilute a rate.

## 4. Six strata are empty by construction

Named on the page rather than silently omitted:

| stratum | why |
|---|---|
| V02 improvised | V02 recorded no improvised material at all |
| V00 and V02 · FM0 | neither has an excluded format or a drifting annotation track |
| V01 and V02 · FM1 | switched off for those vendors |
| V00 · FM4 | switched off for that vendor |

Two more sections are *short* rather than empty — V02 and V03 have only 5 and 1
FM4 files in the whole sample — and the page says so, so a one-clip section does
not read as a sampling accident.

## 5. Twelve defects found by adversarial review, all fixed

Four independent checkers went over the built page — spec compliance, whether the
numbers reconcile against the parquet, whether the prose is supportable, and
integrity plus permissions — and each candidate was then handed to a second agent
whose job was to refute it. 32 candidates were raised and 17 survived refutation;
after removing duplicates across dimensions, twelve distinct defects.

Three were serious enough that the page would have misled a reader:

1. **The four headline tiles printed the caption in the big slot and the number
   in the small grey one.** The renderer unpacked `(number, label)` as
   `(label, number)`. This has been wrong in **every briefing gallery ever
   built**, including the V00 page your PI already has, and nobody caught it —
   including me, twice, on this page.
2. **The five per-check chips disagreed with the table a few centimetres below**:
   FM1 read 5.6% on the chip and 11.3% in the table, FM4 1.0% against 1.3%. The
   chips averaged over every file, the table only over the vendors a check is
   applied to. Both are now the second thing, and each chip names the vendors it
   covers.
3. **Rendered sidecars were shadowing the manifest's clip signals.** The sidecar
   keeps a copy of the signals frozen at render time, and it was winning the
   merge — so any signal added after the clips were rendered vanished silently.
   That is how eight FM4 cards came to read "unknown" where the manifest said
   "empty". A general bug in every builder in this repo, not just this one.

Six more, each smaller:

| | |
|---|---|
| every card printed **both** vendors' FM1 rules, so 118 clips carried a cut point that does not apply to them | now one row, naming the measure that vendor actually uses |
| the eight empty-audio cards showed no FM4 number at all — including V03's only such clip | now an explicit "the released audio file is empty — no samples at all" |
| FM3's prose claimed "5% to 9% throughout" while the table below printed V02 at 4.5% | corrected to 4.5–9.0% |
| the dyad-hours column summed to 782 but the total row said 781 | rounded once, then totalled |
| the participant tile counted bare `participant_id`, which is numbered *within* a vendor, undercounting by 263 | counts (vendor, participant) pairs — 2,489, not 2,226 |
| the transport banner still told the reader to watch the pill that had just been removed | removed |

Three more were in the prose, and these are the ones a professor would actually
have been misled by:

- **The FM1 card quoted the wrong round.** It gave "catches 44 seated files in 45
  where the knee measure caught 41" and attributed it to the 164-label set. Those
  are the **90-label** numbers from `reports/10`, and `reports/12` §3 is titled
  "A number I have now revised twice" precisely because that draw
  over-represented clear cases. The card now gives the settled figures — 52 of 53
  seated caught, 10 of 96 standing discarded — and states the standing cost,
  which the earlier text omitted entirely.
- **The FM0 section blurb gave the wrong reason for every V03 clip in it.** It
  named a broken body fit or a drifting annotation track; all twelve V03 clips
  there are room-camera formats excluded for **far-field audio**, and their body
  fits are fine. A reader would have gone looking for a defect that is not in the
  picture. The blurb now names all three reasons, says which vendor each applies
  to, and warns that the V03 clips will look normal because the problem is what
  you hear.
- **Figures borrowed from earlier rounds were not marked as borrowed**, and one
  had been reworded from "1,379 square-pixel files" to "1,379 usable files" —
  which this sample's own table contradicts, since V01 has 834 files left after
  FM0. Every such figure is now explicitly attributed to the round that produced
  it, and the lede no longer claims the page is self-contained.

## 6. Verified

- 246 videos and 246 download links, **0 broken**; the HTML parses; no dangling
  contents anchor; 4 vendor groups over 22 sections.
- No textarea, radio or rating control: the page collects nothing.
- Directory `0700`, page and every clip `0600`. The 246 entries under `source/`
  are symlinks, whose own mode is never honoured; their targets are `-r--r--r--`
  read-only source data. **No real file is group- or other-writable.**
- Spot-checked renders across three strata: a V00 improvised keep, a repaired V01
  anamorphic clip showing correct proportions, and a V03 file the pipeline calls
  seated in which the participant is plainly seated.
- Headline tiles, detector chips and the rates table all reconcile against the
  parquet, and the dyad-hours column adds to its own total.

Two defects were found during the build and fixed. The sampler could place a clip
one frame past the end of a container that holds one frame fewer than its
annotation grid; it now bounds the start by the container as well. And the
renderer failed the whole clip when that happened, rather than holding the last
frame for a one- or two-frame shortfall — a tolerance that matches the one the
timebase check already uses.

## 7. What the page says is still open

Stated plainly rather than buried: **FM2 rejects 59% to 80% of every vendor** and
is by a wide margin the largest single cost in the pipeline. It is also the only
check no review has ever contradicted. If more hours are wanted, that is the only
place with any to give.
