# Session 2 — corpus measurements and feature validation

Date: 2026-08-05 (America/Los_Angeles)  
Status: **M-1, M-2, M-3, M-4 and the Phase-0 tooling are complete. The Pass-1 and
Pass-2 human reviews require the user and PI and have not run** — see
`reports/02_review_findings.md` and §8.2 below.  
Policy: measurement only; no filtering thresholds, scoring model, or subset
selection decisions

This report distinguishes direct observation from interpretation. Any material
fact not confirmed from the released payload is marked **unverified**.

## 1. Scope and resolved paths

| item | repository path | resolved target | access used in this session |
|---|---|---|---|
| source dataset | `./seamless_interaction` | `/simurgh2/datasets/seamless_interaction` | read-only |
| metadata | `./datasets/seamless_interaction_metadata` | `/simurgh/group/lw29/datasets/seamless_interaction_metadata` | read-only |
| clean-output root | `./seamless_interaction_clean` | `/simurgh2/datasets/seamless_interaction_clean` | writable; zero-byte probe removed after verification |
| neutral SMPL-H model | `./model_files/smplh/SMPLH_NEUTRAL.npz` | `/simurgh/group/lw29/model_files/smplh/SMPLH_NEUTRAL.npz` | read-only; 135,375,142 bytes |

The workspace had been renamed from `seamless_pipeline` to
`seamless_processing` between sessions. Session-1 history is intact in
`.git-session`; its starting HEAD for Session 2 was `8a2da10`. A compatibility
symlink at the former path makes the execution environment's fixed writable
root refer to the renamed repository. This changes no dataset path.

The source's effective protection needs a precise qualification. The IDE
sandbox reports its view as `ro`, but a real Slurm process sees the parent NFS
export mounted `rw`. The dataset directory itself is mode `2755`, owned by
`askhan1:simurgh`; user `lw29` has no effective write access (`test -w` is
false). Production preflights therefore require effective write access to be
false and require every output path to resolve outside the source. They do not
claim that the entire parent export is mounted read-only. No source write probe
was attempted.

## 2. Environment and resource preflight

### 2.1 Clean-output capacity

At observation time, the NFS export containing the clean target reported:

| measurement | observed value |
|---|---:|
| filesystem | NFS |
| block size | 1,048,576 bytes |
| available blocks | 78,032,037 |
| available bytes | 81,822,521,229,312 |
| available capacity | 74.417 TiB (`df -h`: 75T) |
| free inodes | 159,809,610,774 |

This directly refutes the earlier approximately 865 GiB reading for the
resolved clean-output target. Server-side quota limits distinct from `df` are
**unverified**; quota clients are unavailable.

The write check created one zero-byte mode-`0600` file named
`.codex_session2_write_probe` in the clean root, verified its existence, and
removed that exact file. No participant media or model asset was copied.

### 2.2 Runtime

The existing ViBES environment remains usable by absolute interpreter path:

| package | observed version/state |
|---|---|
| Python | 3.10.20 |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |
| Torch | 2.11.0+cu128 |
| `smplx` | import succeeds; package exposes no `__version__` |
| OpenCV | 4.11.0 |

CUDA is unavailable to processes launched outside a Slurm job step. This is an
execution-context fact, not a GPU failure; FK/render workloads are therefore
run only through Slurm.

FFmpeg and ffprobe are `/usr/bin` version `6.1.1-3ubuntu5`. Their complete build
configuration is unchanged from `reports/01_recon.md`; it includes libx264,
AAC, CUDA, OpenCL, and the codecs required by the review renderer.

### 2.3 Slurm state before Session-2 submissions

| item | observed value |
|---|---|
| interactive job | `16499428`, running on `simurgh2` |
| allocation | 16 CPUs, 200 GiB RAM, 2 × L40S GPU |
| allocation end | 2026-08-07 10:59:43 PDT |
| `simurgh` maximum time | 21 days |
| `simurgh-interactive` maximum time | 3 days |
| array cap | 1,001 indices (`MaxArraySize=1001`) |
| association limits | no explicit job, submit, or TRES cap returned by `sacctmgr` |
| partition capacity at check | 640 CPUs total; 132 idle across the reported node groups |

The first M-1 benchmark submission, job `16508930`, exited after two seconds
before probing any participant file because its conservative preflight expected
the parent mount to contain the `ro` option. This exposed the namespace
difference above. The corrected guard checks effective directory permissions
without attempting a write and records the underlying mount honestly.

### 2.4 Outbound network from a batch node

Slurm job `16508832` used one CPU, 512 MiB, and five seconds on batch node
`simurgh1`. DNS resolution and HTTP HEAD requests succeeded without downloading
page bodies or packages:

| endpoint | observed HTTP status |
|---|---:|
| `https://pypi.org/` | 200 |
| `https://huggingface.co/` | 200 |
| `https://repo.anaconda.com/` | 200 |

This verifies outbound access on that batch node at that time. Availability on
every future node remains **unverified**.

### 2.5 Exact preflight commands

```bash
readlink -f seamless_interaction seamless_interaction_clean \
  datasets/seamless_interaction_metadata \
  model_files/smplh/SMPLH_NEUTRAL.npz
df -hT /simurgh2/datasets/seamless_interaction_clean
df -i /simurgh2/datasets/seamless_interaction_clean
stat -f -c \
  'block_size=%S blocks=%b free_blocks=%f avail_blocks=%a files=%c free_files=%d fs_type=%T' \
  seamless_interaction_clean
stat -c '%s %a %n' model_files/smplh/SMPLH_NEUTRAL.npz
findmnt -T /simurgh2/datasets/seamless_interaction -n \
  -o TARGET,SOURCE,FSTYPE,OPTIONS
stat -c '%A %a %U %G %n' \
  /simurgh2/datasets /simurgh2/datasets/seamless_interaction
test ! -w /simurgh2/datasets/seamless_interaction

/simurgh/group/lw29/conda/envs/ViBES/bin/python -c \
  "import sys,numpy,pandas,torch,smplx,cv2; print(sys.version); \
   print(numpy.__version__,pandas.__version__,torch.__version__,cv2.__version__)"
ffmpeg -version
ffprobe -version

scontrol show job 16499428
squeue -u lw29 -o '%.18i %.24P %.28j %.2t %.10M %.10l %.6D %R'
sacctmgr -n -P show assoc user=lw29 account=simurgh \
  format=Cluster,Account,User,Partition,QOS,GrpJobs,MaxJobs,GrpSubmitJobs,MaxSubmitJobs,GrpTRES,MaxTRESPerJob,MaxTRESPU
sinfo -p simurgh,simurgh-interactive -o '%P|%a|%l|%D|%G|%C'

sbatch --parsable slurm/network_probe.sbatch
sacct -j 16508832 --format=JobID,State,Elapsed,ExitCode,NodeList -n -P
```

## 3. Metadata schema preflight

The four tables are now locally available. This section records schema only;
the corpus-wide counts supplied in the Session-2 brief are treated as given
and are not presented as newly derived results.

| table | rows × columns | columns with empty-string values |
|---|---:|---|
| `filelist.csv` | 129,370 × 8 | none |
| `interactions.csv` | 1,312 × 8 | `ipc_b`: 52.210% |
| `participants.csv` | 4,284 × 7 | none (the literal string `Undisclosed` is not null) |
| `relationships.csv` | 5,098 × 4 | none |

`interactions.csv` contains an exported index column named `Unnamed: 0`.
Observed `interaction_type` values are `ipc_conversation` (946 prompt rows),
`grounded_gesture` (322), `collaborative_storytelling` (40), and `charades` (4).
These are prompt-table row counts, not corpus file counts or durations.

## 4. Corpus-wide inventory (M-1)

### 4.1 What was read, and what was not

Two independent passes covered the whole source tree.

| pass | scope | what it opened | result |
|---|---|---|---:|
| catalog | every directory entry under `./seamless_interaction` | `readdir` + `lstat` only | 572,700 files, 39,793,651,185,064 bytes, 863.05 s, 663.6 paths/s |
| headers | the 129,370 `filelist.csv` rows × 4 sibling extensions | `stat` on all four; `ffprobe` on the MP4 container only | 512 shards, all complete |

**No JSON, NPZ, or WAV payload byte was read**, by construction: the scanner
stats those three and probes only the MP4 container. The headers pass ran as a
512-task array at 8 concurrent tasks; wall time per task was 2:00–5:50 and the
whole array finished inside two hours.

Reproduction:

```bash
sbatch slurm/inventory_m1_catalog.sbatch       # job 16509072, 00:14:28
sbatch slurm/inventory_m1_headers.sbatch       # job 16509071, array 0-511%8
sbatch slurm/inventory_m1_summarize.sbatch     # job 16520456, 00:00:06
```

Outputs are git-ignored under `outputs/02_inventory/` (`catalog/files.parquet`,
`header_shards/task_*.parquet`, `summary/*`).

### 4.2 Actual durations per vendor × label × split

This replaces the estimated hour figures. `dyad_hours` is the mean of the two
member durations per interaction, summed; `participant_stream_hours` is the sum
of individual participant video durations and is therefore about twice as large.

| vendor | improvised dev | improvised test | improvised train | naturalistic dev | naturalistic test | naturalistic train | **vendor dyad-hours** |
|---|---:|---:|---:|---:|---:|---:|---:|
| V00 | 10.10 | 5.98 | 974.95 | 17.33 | 13.74 | 463.49 | **1,485.59** |
| V01 | 3.70 | 5.77 | 198.21 | 6.71 | 8.12 | 114.46 | **336.96** |
| V02 | — | — | — | 2.41 | 2.31 | 768.23 | **772.95** |
| V03 | 8.01 | 4.09 | 141.23 | 11.69 | 20.59 | 1,266.85 | **1,452.47** |
| **all** | | | | | | | **4,047.97** |

Measured total participant-stream hours: 8,072.14. Measured dyad-hours:
**4,047.97**, against the brief's given estimate of 4,065 — agreement to 0.4%.

Two figures change the V00 decision in V00's favour:

- **V00 total: 1,485.59 dyad-hours measured** versus the brief's estimated
  ≈1,300. The headroom against a ~300-hour target is larger than assumed.
- **V00 naturalistic-only: 494.56 dyad-hours** (17.33 + 13.74 + 463.49) versus
  the estimated ≈517. Still comfortably above 300 before any gating.

**V02 contains no improvised data at all** — 30,172 files, all naturalistic.
This is a structural fact about the corpus that Session 1's dev-only view could
not show.

Durations come from the MP4 container (`format.duration`), falling back to
stream duration only when the format field is absent. 128,747 of 129,370 rows
have a usable duration; the 623 that do not are itemized in §4.6.

### 4.3 Frame rate and raster by vendor

**V00 is uniformly exact-30 fps at 1080×1920 across the entire corpus.** All
42,826 V00 files with a readable video stream report `r_frame_rate = 30/1`,
`avg_frame_rate = 30/1`, and 1080×1920. The V00 `|avg − r|` disagreement is
exactly 0.0 for every file — mean, p95, p99, and max all zero. Session 1's
dev-only observation is confirmed corpus-wide.

| vendor | distinct (rate, raster) combinations | dominant combination | share |
|---|---:|---|---:|
| V00 | 1 | `30/1` @ 1080×1920 | 100.00% |
| V02 | 1 | `30/1` @ 1080×1920 | 100.00% |
| V03 | 7 | `30000/1001` @ 2160×3840 | 96.48% |
| V01 | 12 | `30000/1001` @ 1080×1920 | 36.64% |

V01 is genuinely heterogeneous: 1080×1920, 2160×2160, 1920×1080, 1012×1920 and
rates including `30/1`, `30000/1001`, `45000/1501`, `48000/1001`, and three
non-reducible ratios such as `2048074742/45621735`. V03 adds 3840×2160,
640×480, 1080×960, and a 2180×3840 raster.

Files where the measured average rate departs from the container rate by more
than 0.01 fps: **416 total — 273 in V01 (max |Δ| = 6.82 fps) and 143 in V03
(max 0.05 fps); zero in V00 and V02.** This is the concrete reason the brief's
"always use each file's measured `avg_frame_rate`" rule is not optional.

Embedded MP4 audio: V00 and V03 are uniformly 48 kHz stereo; V02 is 48 kHz mono
on 30,052 files with 52 stereo and 19 at 44.1 kHz; V01 mixes 4,013 mono and
7,921 stereo. 128,724 of the 128,747 readable-video files carry an embedded
audio stream.

### 4.4 Participants and per-participant volume

**`participant_id` alone is not a participant identity.** `participants.csv`
has 4,284 rows and only 3,599 distinct `participant_id` values, so the table is
keyed on `(vendor_id, participant_id)`. In the file inventory, 627 participant
IDs appear under more than one vendor (V02+V03 236, V01+V03 226, V00+V01 81,
V00+V01+V03 50, V00+V03 34). Counting raw `participant_id` yields 3,630 where
the correct pair count is **4,307** — a 16% undercount. Every diversity floor
must be stated over `(vendor, participant_id)` pairs.

| vendor | distinct participants | participant-stream hours | median h/participant | p95 | max | largest single share |
|---|---:|---:|---:|---:|---:|---:|
| V00 | 1,316 | 2,964.09 | 0.814 | 14.99 | 30.30 | 1.02% |
| V01 | 706 | 671.14 | 0.741 | 2.30 | 9.65 | 1.44% |
| V02 | 540 | 1,545.46 | 1.483 | 8.57 | 10.16 | 0.66% |
| V03 | 1,745 | 2,891.45 | 0.979 | 5.24 | 35.60 | 1.23% |

The distribution is heavily skewed: the median participant contributes 0.87 h
while p99 is 20.09 h and the maximum is 35.60 h. A per-participant duration cap
in the 1–2% range is therefore load-bearing, not cosmetic — the top V00
participant alone already holds 1.02% of V00's hours.

2,700 file rows (2.1%) have no matching `participants.csv` row and 3,040 (2.4%)
have no matching `relationships.csv` row. All 129,370 rows join to
`interactions.csv` on `interaction_id = prompt_hash`.

A caution on within-V00 identity: 51 V00 numeric stems carry both a bare and a
letter-suffixed variant (for example `0602` and `0602A`), which the file
inventory treats as two participants. Whether these are the same human is
**unverified**.

### 4.5 Activity type, with hours

The join key is `file_id`'s `interaction_id` field against
`interactions.csv:prompt_hash`; all rows match. These are measured hours, not
the prompt-table row counts quoted in §3.

| interaction_type | files | dyad-hours | share | V00 dyad-hours |
|---|---:|---:|---:|---:|
| `ipc_conversation` | 107,749 | 3,448.21 | 85.18% | 1,321.24 |
| `grounded_gesture` | 12,681 | 377.44 | 9.32% | 83.48 |
| `collaborative_storytelling` | 5,751 | 147.13 | 3.64% | 39.20 |
| `charades` | 3,189 | 75.19 | 1.86% | 41.68 |

The two activities the brief singles out:

- **`charades` — pure visual communication, to be excluded from
  speech-conditioned training: 75.19 dyad-hours corpus-wide (1.86%), of which
  41.68 are V00.** Excluding it costs V00 2.8% of its hours.
- **`grounded_gesture` — the language-grounded, deliberately gesture-dense and
  speech-aligned game: 377.44 dyad-hours corpus-wide (9.32%), of which 83.48
  are V00** (26.11 improvised train, 53.50 naturalistic train, the rest dev and
  test). This is the highest-value-per-hour pool in the corpus for this project
  and is large enough to matter on its own.

By label, `grounded_gesture` is 330.55 naturalistic against 46.89 improvised
dyad-hours, so the gesture-dense activity is concentrated in exactly the split
the PI prefers.

### 4.6 Empty placeholders and unreadable media — the size signature is insufficient

The brief's detector (≈261-byte MP4 with ≈58-byte WAV) finds **285** bundles.
`ffprobe` independently reports **439** files with no video stream. Every one of
the 285 is inside the 439, so the size rule produces no false positives — but it
**misses 154 files, and in particular misses all 106 empty V00 bundles.**

Distinct observed signatures among the 439:

| mp4 bytes | wav bytes | npz bytes | json bytes | files | vendors | reading |
|---:|---:|---:|---:|---:|---|---|
| 261 | 58 | 1,981 | 96 | 285 | V01, V02, V03 | the brief's signature |
| 708 | 58 | 5,305 | 96 or 97 | **106** | **V00 only** | a second, larger placeholder the 261-byte rule cannot see |
| 261 | 19.9 MB – 264 MB | 1,981 | 7.9 KB – 415 KB | 39 | V03 | placeholder video with **real audio and real transcript** |
| 1.40 MB – 1.87 MB | 30 MB – 40 MB | 1,981 | 31 KB – 96 KB | 9 | V02 | multi-megabyte MP4 that still carries no video stream |

Two further classes are invisible to any size check:

- **7 V03 files are truncated MP4 containers** — `moov atom not found`, with
  file sizes from 65.7 MB to 286.2 MB. These look entirely healthy by size and
  fail only when a decoder opens them.
- **187 files have a readable video stream but a 58-byte (empty) WAV** — 64 in
  V01, 64 in V02, 59 in V03, **zero in V00**. For speech-conditioned training
  this is a silent-audio failure that no video-side check would catch.

Recommendation: the corpus-wide emptiness detector must be
`ffprobe`-status-based (no video stream) unioned with a WAV-size check, not a
size signature. Cost is acceptable — this is exactly what M-1 already measures.

Counting all four classes, V00 contributes 106 unreadable bundles out of 42,932
(0.247%) and no silent-audio, no truncated-container, and no
placeholder-with-real-audio cases at all.

### 4.7 Bytes, measured

| scope | files | bytes | TiB |
|---|---:|---:|---:|
| every physical file in the source, including archives | 572,700 | 39,793,651,185,064 | **36.192** |
| `filelist`-expected extracted payloads present locally | 516,748 | 24,954,264,130,461 | **22.696** |

The extrapolated "~20–25 TiB" is replaced by a measurement: **22.696 TiB of
extracted participant payload**, inside a 36.192 TiB source tree.

By extension, within the expected extracted payload:

| extension | files | TiB | share of payload bytes |
|---|---:|---:|---:|
| `.mp4` | 129,193 | 15.890 | 70.01% |
| `.wav` | 129,181 | 5.069 | 22.34% |
| `.npz` | 129,181 | 1.728 | 7.61% |
| `.json` | 129,193 | 0.0085 | 0.04% |

**MP4 + WAV are 92.35% of the payload bytes.** The brief's "~91%" is confirmed,
which is what justifies the Stage-1 rule that no MP4 or WAV may be opened: every
Stage-1 signal lives in the remaining 7.65%.

The 13.496 TiB difference between the two scopes is 13,200 `.tar` archives, plus
21,383 zero-byte `.lock` files, 21,349 `.metadata` files (2.66 MB total), and 19
zero-byte `.incomplete` files. The last three are Hugging Face download
bookkeeping, not dataset content. Session 1's observation that improvised/dev
retains adjacent tars generalizes: tars exist corpus-wide and double-counting
them would inflate any storage estimate by 59%.

### 4.8 Coverage in both directions, and the 61-file dev gap

| check | result |
|---|---:|
| `filelist.csv` rows | 129,370 |
| rows with all four modalities present locally | 129,181 |
| rows with at least one modality absent | 189 |
| absent expected payload paths | 732 |
| **local payload files not expected by `filelist.csv`** | **0** |

Coverage is exact in the reverse direction: every `.json`/`.mp4`/`.npz`/`.wav`
file in the source tree corresponds to a `filelist.csv` row. Nothing is staged
that the metadata does not describe.

Of the 189 incomplete bundles, 177 have **no** modality present at all and 12
have `.json` + `.mp4` but no `.npz` + `.wav`. They fall in V02 naturalistic dev
(9), V03 naturalistic dev (52), V03 naturalistic test (94), and V03
naturalistic train (34).

**The 61-file dev gap is fully explained.** `filelist.csv` lists 2,180 dev rows;
2,119 have all four modalities locally; 61 do not — **9 in V02 naturalistic dev
and 52 in V03 naturalistic dev**, all of them in the "no modality present"
class. The staged dev tree is not missing files relative to what it contains; 61
bundles were simply never extracted. There is no V00 or V01 dev gap.

### 4.9 Worker-scaling benchmark

`ffprobe` header throughput on deterministic disjoint file sets per worker
setting, seed 20260805, 256 files each. Caches were not flushed (that requires
privileges we do not have), so this is "cold-ish", and the disjoint sets are
what stop a warm repeat from dominating.

| workers | files/s | speed-up vs 1 | per-worker efficiency | wall for 256 files |
|---:|---:|---:|---:|---:|
| 1 | 2.52 | 1.00× | 100% | 101.40 s |
| 4 | 7.98 | 3.16× | 79% | 32.08 s |
| 8 | 19.86 | 7.87× | 98% | 12.89 s |
| 16 | 31.28 | 12.39× | 77% | 8.18 s |

Scaling is close to linear to 8 workers and still gains at 16. The associated
`logical_mp4_mib_per_s` column (337 → 4,296 MiB/s) divides represented file
sizes by wall time; it is **not** physical bytes read, because `ffprobe` reads
only headers. It is recorded for continuity with Session 1 and must not be used
as a throughput budget.

Grounded Stage-0 budget: at the observed 31.28 files/s for 16 workers, one
16-worker task covers 129,370 header probes in 68.9 minutes. The realized
512-task × 2-worker array did the same work in under two hours of wall time at 8
concurrent tasks, which is the number the pipeline spec uses.

### 4.9b NPZ payload read throughput — the number that actually bounds Stage 1

The header benchmark says nothing about payload throughput. Stage 1 reads whole
NPZ members (Session 1 established that DEFLATE-compressed members cannot be
true-memory-mapped, so there is no partial-read shortcut) plus the small JSON.
`scripts/benchmark_npz_read.py` measures exactly that array set — no MP4, no WAV.

The first run of this benchmark accidentally produced the most useful result in
it. Re-reading the same seed's files gave 253–1,549 MiB/s; reading a disjoint,
never-touched set gave 17.8–168.4 MiB/s. **Cache state changes the answer by
roughly a factor of nine**, exactly as Session 1 warned. Both are reported
because using the warm number would understate the Stage-1 budget by 9×.

| workers | cold-ish files/s | cold-ish MiB/s | warm files/s | warm MiB/s |
|---:|---:|---:|---:|---:|
| 1 | 1.09 | 17.8 | 18.30 | 253.5 |
| 4 | 6.37 | 86.0 | 72.66 | 994.5 |
| 8 | 7.81 | 107.1 | 117.79 | 1,546.5 |
| 16 | **11.26** | **168.4** | 115.91 | 1,548.7 |

Cold-ish scaling is strongly sub-linear: 16 workers deliver 9.5× one worker, and
going 8 → 16 buys only +57%. A single node's NFS path saturates near **170
MiB/s** for this access pattern. The warm numbers plateau at 8 workers, which is
consistent with CPU-bound DEFLATE decompression once the bytes are local.

Grounded Stage-1 budget, using the cold-ish 16-worker figure:

| scope | bytes | wall-clock, one 16-worker node |
|---|---:|---:|
| corpus-wide (129,370 files) | 1.7365 TiB | **3.00 h** |
| V00 only (42,932 files) | 0.9771 TiB | **1.69 h** |

Cross-check by file count: 129,370 ÷ 11.26 files/s = 3.19 h, agreeing with the
byte-based figure to within 6%.

**Aggregate throughput across several nodes is unverified.** The benchmark used
threads on one node, so the honest reading is that Stage 1 costs about three
node-hours corpus-wide and whether running it on eight nodes at once gives 8×
depends on server-side NFS capacity that was not measured.

```bash
python scripts/benchmark_npz_read.py --seed 20260807 \
  --out outputs/02_inventory/benchmark/npz_read_scaling_coldish.json
```

### 4.10 What M-1 changed

1. The V00 volume question is settled with measurements, and in V00's favour:
   1,485.59 dyad-hours total, 494.56 naturalistic-only.
2. V00's uniform exact-30 / 1080×1920 timing holds corpus-wide, so choosing V00
   removes the mixed-rate problem instead of managing it.
3. The brief's placeholder size signature misses 154 files including all 106
   empty V00 bundles; emptiness detection must be `ffprobe`-based.
4. 187 files pair readable video with an empty WAV, and 7 V03 MP4s are truncated
   containers. Neither class is detectable by size.
5. `participant_id` is only unique within a vendor; diversity quotas must key on
   `(vendor, participant_id)`.
6. Storage is measured at 22.696 TiB of payload with MP4+WAV at 92.35%, which
   confirms the Stage-1 no-media rule.
7. The 61-file dev gap is 9 V02 + 52 V03 naturalistic dev bundles that were
   never extracted; there is no reverse-direction coverage problem at all.

## 5. Camera-parameter tests

### 5.1 Bounded sample and explicit hypothesis

M-2 and M-4 used a fixed seven-file dev manifest spanning all four vendors,
with 64 deterministic valid frames per file (448 FK frames). Camera summary
statistics use all 42,115 jointly SMPL-H/box-valid frames in those files. The
seed is `20260805`; file IDs are frozen in
`configs/smplh_validation_sample.csv`.

The tested camera is a project hypothesis, not intrinsics carried in the
payload:

```text
depth numerator = 2 * 5000 / 256 = 39.0625
f_px = 5000 / 256 * max(native width, native height)
principal point = native-raster centre
```

### 5.2 M-2 observations

| measurement | observed value |
|---|---:|
| valid frames | 42,115 |
| `tz` median / p05 / p95 | 37.771 / 31.656 / 55.816 |
| `tz × box_height / frame_height` median | 31.831 |
| within-file robust CV of that product, median / max | 1.489% / 3.895% |
| implied `s = 39.0625/tz` vs normalized box height, pooled Pearson *r* | 0.7359 |
| same relationship across seven file medians | 0.6003 |
| `tx` vs normalized box-centre x, pooled *r* | 0.9329 |
| `ty` vs normalized box-centre y, pooled *r* | 0.9634 |
| `global_orient[0]` median | 3.1476 rad |
| median `abs(abs(global_orient[0]) - pi)` | 0.0800 rad |

The full-frame scaled-focal projection is decisively better than the two
explicit alternatives on the 23 body/foot landmarks:

| camera candidate | median error | p95 error |
|---|---:|---:|
| scaled focal + raster-centre principal point | 31.28 px | 244.45 px |
| unscaled 5000-px focal + raster centre | 550.50 px | 1,704.41 px |
| tested bounding-box crop transform | 148.09 px | 746.17 px |

Together with the reprojection result below, these observations support the
hypothesis that `smplh:translation` is an HMR-style camera/tracker parameter,
not physical participant root motion. This is an inference from several tests,
not metadata encoded in the file. The project recommendation is therefore to
remove it from motion features, retain its temporal stability as a tracking
quality signal, and rename the Session-1 provisional acceleration accordingly.

The approximately-pi first orientation component is consistent with an x-axis
flip between body-model and image conventions. It does not by itself identify
a world coordinate frame.

## 6. SMPL-H forward-kinematics validation

### 6.1 Fixed model choices

The accepted run used the requested neutral model in place, without copying or
modifying it:

```python
smplx.create("./model_files", model_type="smplh",
             gender="neutral", ext="npz",
             use_pca=False, flat_hand_mean=SETTING,
             num_betas=16, batch_size=16)
```

All 16 betas are zero by project convention; no beta values are observed in
the released payload. Full `(N, 15, 3)` axis-angle hand arrays are supplied
directly (`use_pca=False`). Both `flat_hand_mean` settings were run on the same
frames.

### 6.2 Reprojection acceptance result

| joint group / setting | points | median px | p95 px | median / p95 as box-height fraction |
|---|---:|---:|---:|---:|
| body 17, either hand-mean setting | 7,616 | 25.53 | 103.74 | 1.409% / 4.931% |
| all 42 hand keypoints, `flat_hand_mean=False` | 18,816 | 53.29 | 130.99 | 2.877% / 7.071% |
| all 42 hand keypoints, `flat_hand_mean=True` | 18,816 | **39.49** | **94.96** | **2.172% / 4.854%** |
| body 23 including six foot landmarks | 10,304 | 31.28 | 244.45 | 1.748% / 12.822% |

`flat_hand_mean=True` lowered the all-hands median by 25.9% and p95 by
27.5%, and had the lower per-file hand median in 7/7 files. It is therefore the
empirically selected project setting.

The acceptance criterion **passes for body 17 and both hands**: error is small
relative to the person box and the tested camera alternatives fail by much
larger margins. Errors are not wholly unstructured: the 23-landmark p95 is
dominated by extreme foot-landmark errors (maximum 3,355 px), while body 17 is
substantially tighter. Foot mapping/provenance remains **unverified** and foot
extras are excluded from the acceptance claim and from the upper-body renderer.

Subtracting the output pelvis yielded an exact zero pelvis in the sampled FK
arrays. The neutral SMPL-H model returns metres; root-relative joints are
therefore multiplied by 1,000 for metric millimetres. This unblocks genuine
root-relative joint acceleration and wrist-speed measurements. It does not
make the HMR camera translation physical.

Numeric outputs, including per-point errors, are private/generated and
git-ignored under:

```text
outputs/session2/smplh_validation_tiny/
```

The acceptance run was repeated for a resource record and is stamped with
clean Git SHA `da3892d3a6d8d77c5234e2b549cf9316634a78a9`,
config SHA-256
`a01adc4d5c7ad3d8f1bc065ff95913dd30007fb3608bee1ed7133bac7dc4a5c5`,
Slurm job `16499428`, step `3`, four CPUs, and 32 GiB requested memory. The
Python payload took 5.60 s wall time and 825,216 KiB maximum RSS. Its numeric
results exactly reproduced the original accepted step.

### 6.3 Reproduction commands

```bash
PYTHONPATH=src /simurgh/group/lw29/conda/envs/ViBES/bin/python -m pytest -q \
  tests/test_smplh_fk.py

srun --jobid=16499428 --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=32G bash -lc \
  'cd /afs/cs.stanford.edu/u/lw29/seamless_processing && \
   /usr/bin/time -v \
     -o outputs/session2/smplh_validation_tiny/resource_usage.txt \
     env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
     /simurgh/group/lw29/conda/envs/ViBES/bin/python \
       scripts/validate_smplh_reprojection.py \
       --config configs/smplh_validation.yaml'
```

## 7. Dev feature characterization (M-3)

Distributions and correlations only. No threshold is applied and nothing is
selected. The single parameter that needs a number — the rest-pose displacement,
100 mm — is reported as a parameter alongside the threshold-free percentiles it
was derived from.

### 7.1 Scope and unit of measurement

| item | value |
|---|---|
| manifest | `configs/dev_sample.csv` (120 Session-1 dev files) |
| unit | 150-frame span (5 s at the release's own annotation step) |
| spans per file | 8, evenly spaced including the first and last usable span |
| spans measured | **928** across **116** files |
| files skipped | 2, both `no_video_stream` (recorded, not silently dropped) |
| signal columns | 283 numeric measurements per span |
| FK | neutral SMPL-H, β = 0, `use_pca=False`, `flat_hand_mean=True`, translation zeroed, pelvis subtracted, m × 1000 |
| timing | each file's measured `avg_frame_rate`; never assumed 30 |

Spans are chosen on a fixed stride and are **not** steered towards valid frames.
A characterization that skipped failures would describe a corpus we do not have.

```bash
sbatch slurm/m3_dev_features.sbatch     # job 16520898, 00:05:01, 1 GPU
python scripts/m3_analyze.py
```

### 7.2 Normalized 2D kinematics by joint group

Released keypoints are divided by `max(width, height)` **before anything else**,
so a 3840×2160 file and a 1080×1920 file are comparable. A single isotropic
divisor is used rather than dividing x by width and y by height, because
per-axis division makes a diagonal motion's speed depend on its direction.
Box-height-normalized variants are reported alongside, since raster
normalization still leaves a person twice as far from the camera moving half as
fast.

Median-of-span values, reported as p5 / p50 / p95 across the 928 spans, in
raster-fractions per second:

| joint group | speed p5 | speed p50 | speed p95 | high-pass jitter p5 | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| body 17 | 0.0083 | 0.0477 | 0.1238 | 0.0003 | 0.0016 | 0.0027 |
| feet 6 | 0.0102 | 0.0565 | 0.1403 | 0.0004 | 0.0019 | 0.0033 |
| face 68 | 0.0086 | 0.0481 | 0.1272 | 0.0003 | 0.0017 | 0.0026 |
| left hand | 0.0080 | 0.0658 | 0.2489 | 0.0003 | 0.0021 | 0.0042 |
| right hand | 0.0083 | 0.0660 | 0.2552 | 0.0003 | 0.0022 | 0.0045 |

Hands move roughly 38% faster than the body at the median and have 3.5× the
spread at p95 — the dynamic range that matters for gesture is concentrated
exactly where tracking is hardest.

Two jitter definitions are computed deliberately: `accel2d` is the plain
second-difference magnitude, and `jitter2d_hp` is the residual after subtracting
a centred 5-frame moving average. §7.5 is why both exist.

### 7.3 Geodesic angular velocity, and the size of the error it avoids

Rotation composition (`R_t^T R_{t+1}`, angle from the trace) throughout. Never
axis-angle subtraction. Values are rad/s, reported as p5 / p50 / p95 across
spans:

| measurement | p5 | p50 | p95 |
|---|---:|---:|---:|
| root geodesic rate, span p50 | 0.024 | 0.070 | 0.216 |
| body geodesic rate, span p50 | 0.027 | 0.081 | 0.221 |
| body geodesic rate, span p99 | 0.265 | 1.911 | 5.899 |
| left-hand geodesic rate, span p50 | 0.105 | 0.324 | 1.041 |
| right-hand geodesic rate, span p50 | 0.106 | 0.304 | 1.042 |
| **naive axis-angle delta, span p99** | 0.259 | 2.074 | 6.979 |
| **naive axis-angle delta, span max** | 0.519 | **8.009** | **188.335** |

Session 1's warning is now quantified corpus-of-dev-wide rather than anecdotally:

- **404 of 928 spans (43.5%) contain at least one naive axis-angle delta above
  100 rad/s.** These are wrap artifacts, not motion.
- The ratio of naive max to geodesic p99 has median 3.7, p95 112, and **maximum
  813×**.

Any velocity or acceleration feature built on raw axis-angle differences would be
dominated by these artifacts. The geodesic form is not a refinement; it is a
correctness requirement.

### 7.4 Metric 3D motion — the first genuine millimetres

With M-4 passed, FK output with translation zeroed and the pelvis subtracted
gives real millimetres. p5 / p50 / p95 across spans:

| measurement | p5 | p50 | p95 |
|---|---:|---:|---:|
| body accel, span p50 (mm/frame²) | 0.212 | **0.519** | 1.215 |
| body accel, span p95 (mm/frame²) | 1.062 | 2.576 | 6.973 |
| left-hand accel, span p50 (mm/frame²) | 0.429 | **1.270** | 5.830 |
| right-hand accel, span p50 (mm/frame²) | 0.412 | 1.290 | 6.152 |
| left-wrist speed, span p50 (mm/s) | 12.5 | **41.2** | 255.4 |
| either-wrist speed, span p90 (mm/s) | 28.1 | 139.6 | 917.1 |
| left-wrist displacement from its own median, span p95 (mm) | 5.2 | 50.7 | 422.9 |

Hand acceleration is 2.4× body acceleration at the median, and its p95 is 4.8×
the body's — hands are both faster and far more variable.

This also settles what Session 1's provisional column was measuring. Session 1's
`accel_mm_per_frame2`, built from root translation, had median 119.8 nominal
mm/frame². The genuine root-relative body figure is **0.519 mm/frame²**, about
230× smaller. The Session-1 column was measuring tracker/camera motion, which is
exactly the M-2 conclusion arrived at independently. Renaming it is not cosmetic.

**Gesture-activity references** used throughout §7.5: `speed2d_body17` p90 (2D,
available on every file), `wrist3d_either_speed_mm_per_s` p90 (metric), and
`angvel_left_hand` p90 (rotational).

`rest_exit_frac` at the 100 mm parameter has p5/p50/p95 = 0.00 / 0.00 / 0.62 —
more than half of all spans have both wrists within 100 mm of their own median
position for the whole 5 s. Gesture is bursty, and a large fraction of any
recording is rest.

The parameter-free `wrist_above_hip_frac` proxy turned out **near-binary and
therefore weak**: 603 of 912 spans are exactly 1.0 and 101 are exactly 0.0, with
only 208 in between. It is recorded but should not be used as a graded activity
measure; the displacement percentiles are the usable form.

### 7.5 The central negative result: jitter measures motion, not noise

This is the finding that most changes the plan.

Of 135 candidate quality signals, **106 (78.5%) reach |ρ| ≥ 0.50 against at
least one gesture-activity reference.** Every jitter and acceleration variant is
in that set:

| quality signal | ρ vs 2D wrist speed | ρ vs 3D wrist speed | ρ vs hand angular rate | ρ after residualizing on 3D wrist speed |
|---|---:|---:|---:|---:|
| `jitter2d_hp_body17_rasterfrac_p90` | **0.903** | 0.635 | 0.492 | 0.045 |
| `accel3d_left_hand_mm_per_frame2_mean` | 0.502 | 0.806 | **0.903** | 0.140 |
| `accel2d_body17_rasterfrac_per_s2_p75` | 0.880 | 0.697 | 0.563 | 0.056 |
| `jitter2d_hp_left_hand_rasterfrac_p50` | 0.689 | 0.820 | 0.760 | 0.062 |
| `jitter2d_hp_body17_rasterfrac_p50` | 0.745 | 0.751 | 0.637 | 0.051 |
| `hand_left_pose_temporal_var_mean` | 0.407 | 0.704 | 0.842 | 0.056 |
| `accel3d_body_mm_per_frame2_p50` | 0.638 | 0.789 | 0.624 | 0.046 |

Box-height normalization does not fix it: `jitter2d_hp_body17_boxheightfrac_p50`
still correlates at 0.750. Neither does the high-pass form, which was designed
to suppress smooth trajectories — it is the *most* confounded signal in the set
at 0.903.

**Consequence: used as a hard gate, any of these signals would preferentially
reject the most animated speakers — precisely the data a co-speech gesture model
needs most.** This is a bias mechanism, not a quality filter.

Residualizing on rank-transformed 3D wrist speed removes the confound: every
flagged signal drops to |ρ| ≤ 0.15, most to ≤ 0.06. The recommendation is
therefore that no raw jitter or acceleration statistic may act as a gate;
residualized variants may rank, and only after the human review shows they track
something reviewers actually dislike.

Signals that are **not** activity-confounded, and are therefore usable as
quality signals on their own terms:

| signal | max |ρ| vs activity |
|---|---:|
| `valid_frac_box` | 0.035 |
| `hand_*_usable_keypoint_frac` | 0.035 |
| `hand_left_pose_frozen_step_frac` | 0.183 |
| `hand_left_out_of_frame_point_frac` | 0.306 |
| `valid_frac_smplh` / `_guarded` / `smplh_and_box` | 0.353 / 0.355 / 0.353 |
| `valid_frac_movement` | 0.383 |
| `speaking_frac` | 0.398 |

Note that `valid_frac_smplh` itself correlates at −0.35 with activity: the
SMPL-H mask is *modestly* more likely to be invalid when the participant is
moving more. Even the released masks are not activity-neutral, and a gate on
them carries a small version of the same bias.

### 7.6 Hands are first-class — and one hand signal turns out to be redundant

**Per-hand availability from released 2D keypoints is exactly identical to
`valid_frac_box` on all 928 spans.** Not correlated — identical, to floating
point, for `hand_left_usable_keypoint_frac`, `hand_right_usable_keypoint_frac`,
and both `complete_frame_frac` columns. Zero spans differ.

The reason is the zero-fill behaviour Session 1 observed: a box-invalid frame
zero-fills all 133 keypoints, and a box-valid frame always carries a complete
hand. The release therefore **never marks a hand as independently missing.**

This has two consequences that pull in opposite directions and both matter:

1. Session 1's blanked-hand test proved that no signal catches a blanked hand.
   M-3 shows the reason is partly that **the failure mode does not occur in the
   released dev data** — there is nothing to catch. The hand-availability column
   should be kept as an explicit guard, but it must be reported as *currently
   redundant with box validity*, not as new coverage.
2. The failure mode the PI actually named is **unnatural hand motion**, which is
   a pose-plausibility problem, not an availability problem. The hand pose is
   always present. Whether it is *correct* is untested by anything we compute.
   **We currently have no signal for an implausible-but-present hand pose.**

Two hand-pose signals do carry independent information:

- `hand_pose_norm_median_rad` has p5/p50/p95 = 1.78 / 2.72 / 3.86. With
  `flat_hand_mean=True` a zero vector is a flat hand, so typical hands are
  strongly articulated and far from default. **`pose_exact_zero_frac` is 0.0 on
  every span** — near-default hands simply do not appear in dev.
- `hand_pose_frozen_step_frac` (consecutive identical 45-parameter vectors)
  correlates with SMPL-H invalidity at ρ = 0.575 left and 0.629 right, while
  being nearly activity-independent (0.183).

That second signal refines a Session-1 open question. Session 1 found SMPL-H
invalid pose to be "finite, nonzero, and usually varying — not a simple
last-valid-frame hold." Measured per modality, the **hand** pose behaves
differently from the body:

| span `valid_frac_smplh` | spans | median frozen-step frac (L) | (R) | median hand-pose temporal variance (L) |
|---|---:|---:|---:|---:|
| exactly 0 | 16 | **0.221** | **0.242** | 0.00002 |
| 0 – 0.5 | 19 | 0.013 | 0.060 | 0.00803 |
| 0.5 – 1 | 83 | 0.000 | 0.000 | 0.02388 |
| exactly 1 | 810 | 0.000 | 0.000 | 0.00714 |

In fully SMPL-H-invalid spans the hand pose is frozen for a fifth of steps at the
median and up to **99.3%** in the worst observed span, with temporal variance
collapsing by three orders of magnitude. On fully valid spans it is exactly zero.
So SMPL-H-invalid regions do partially hold the hand configuration even where the
body pose varies. The precise upstream cause remains **unverified**.

Hand out-of-frame fraction is negligible in dev: p95 = 0.0029 of placed points.

### 7.7 Speaking fraction

From the embedded `metadata:vad` intervals, merged before measurement so
duplicate annotation cannot push the fraction above one, and measured **over each
span's own 5-second window**. An earlier version of this measurement computed the
fraction once per file and copied it into all eight span rows; that made the
distribution file-weighted while claiming 928 spans and attenuated the
correlation against span-local signals. Both figures are reported below because
the difference between them is itself informative.

| measurement | p5 | p50 | p95 |
|---|---:|---:|---:|
| `speaking_frac`, span-local | 0.000 | **0.109** | 0.925 |
| `speaking_frac_file`, whole file | 0.000 | 0.322 | 0.601 |
| VAD segments intersecting the span | 0 | 1 | 4 |
| median VAD segment length, s (spans with speech) | 0.444 | 1.548 | 7.196 |
| speech seconds inside the span | 0.000 | 0.547 | 4.628 |
| spans where the annotation overruns the media | **0** | | |

**415 of 928 spans (44.7%) contain no speech at all**, against only 8 of 116
files whose whole recording is silent. Speech is bursty at the five-second scale:
the file-level median of 0.322 is an average over spans that are mostly either
near-silent or near-continuous, and 17 spans are 100% speech. For a
speech-conditioned model this is a first-order fact about window selection — a
naive uniform windowing would draw a plurality of windows with no conditioning
signal in them.

Span-local speaking fraction correlates with gesture activity at ρ = 0.398
against 3D wrist speed and 0.247 against 2D. That is moderate rather than weak,
and it is materially higher than the ρ = 0.219 the file-level version gave, which
is exactly the attenuation expected from pairing a file-level number against a
span-local one. People do gesture more while speaking; speaking fraction is a
partially independent axis, not an orthogonal one.

The VAD annotation never extends past the media duration on any span, so the
audio and annotation grids agree at this resolution.

### 7.8 What M-3 changed

1. **Jitter and acceleration, in every variant tested, measure motion rather
   than noise** (|ρ| up to 0.903 with gesture activity). None may be a gate.
   Residualizing on 3D wrist speed fixes the confound to |ρ| ≤ 0.15.
2. **Naive axis-angle deltas exceed 100 rad/s on 43.5% of spans.** Geodesic
   composition is mandatory, not preferred.
3. **Metric FK works**: body 0.519 mm/frame², hands 1.270 mm/frame², wrists 41
   mm/s at the median. Session 1's provisional translation column was 230× larger
   and was measuring the camera.
4. **Per-hand availability is exactly `valid_frac_box`** — the release never
   drops a hand independently, so that column adds no coverage today.
5. **Nothing we compute detects an implausible-but-present hand pose**, which is
   one of the PI's four named failure modes. Frozen-hand detection catches the
   held-pose case only.
6. SMPL-H-invalid regions freeze the hand pose far more than the body pose,
   refining a Session-1 open question.
7. Even the released validity masks are mildly activity-correlated (ρ ≈ −0.35),
   so gating on them is not bias-free either.
8. **44.7% of five-second dev spans contain no speech at all**, against 6.9% of
   whole files being silent. Speech is bursty at the window scale, so window
   selection for a speech-conditioned model cannot be uniform over time.

## 8. Closing

### 8.1 What the measurements changed about the plan

1. **V00 is now the recommendation on measured grounds, not a hunch.** 1,485.59
   dyad-hours total and 469.46 naturalistic-only after excluding charades and
   sub-30-second files, against a ~300-hour target — and it is the only vendor
   with uniform exact-30 / 1080×1920 timing corpus-wide, zero empty-WAV files,
   zero truncated containers, and the best dev span validity (91.96% fully valid
   guarded, against V03's 76.89%). See `docs/pipeline_spec_v1.md` §8.
2. **Jitter and acceleration are disqualified as gates.** Every variant measures
   gesture activity (up to *r*<sub>s</sub> = 0.903), so gating on them would
   preferentially discard the most animated speakers. Residualizing on
   rank-transformed 3D wrist speed removes the confound arithmetically
   (|*r*<sub>s</sub>| ≤ 0.15) but has no human validation yet.
3. **The gate list shrank to integrity checks plus one content gate.** G1–G3 and
   G5 remove unreadable, silent, incomplete, or too-short files; G4 removes
   `charades` because the model is speech-conditioned. Nothing else gates. All
   validity fractions rank only, partly because `valid_frac_smplh` is itself
   activity-correlated at −0.35.
4. **Emptiness detection changed method.** The size signature misses 154 files
   including all 106 empty V00 bundles; detection is now `ffprobe`-status based,
   unioned with a WAV-size check.
5. **Diversity quotas re-keyed.** `(vendor_id, participant_id)`, because 627 bare
   IDs collide across vendors. The per-participant cap is stated relative to the
   shipped subset, not the pool: at the pool scale a 1% cap binds on 2
   participants and means nothing; at a 300-dyad-hour target it binds on 91.
6. **Metric features are unblocked and one column is renamed.** FK gives genuine
   millimetres (body 0.519 mm/frame², hands 1.270, wrists 41.2 mm/s). Session 1's
   `accel_mm_per_frame2` was 230× larger and is renamed
   `camera_translation_accel_nominal`, reclassified from motion to tracking
   quality.
7. **Storage is a non-issue and the no-media output is confirmed.** 22.696 TiB of
   payload measured, MP4+WAV 92.35% of it; a 300-dyad-hour no-media subset is
   ≈ 77 GiB, 0.10% of the 74.417 TiB available.
8. **Two of the PI's four named failure modes have no detector at all** —
   implausible-but-present hand pose, and within-file audio/video desync. A third
   (noisy body pose) has only a confounded detector. This is the most important
   result of the session and is stated as a first-class finding in
   `docs/pipeline_spec_v1.md` §3, not as a caveat.

### 8.2 Decisions needed from you

1. **Run the Pass-1 exploratory review.** Open
   `artifacts/private_review_session2/pass1_exploratory.html` over SSH
   port-forward or VS Code Remote, watch the 100 clips, write free text, export
   the JSON. Everything downstream of it is blocked, by design.
2. **Vendor strategy.** V00 naturalistic-only is the recommendation: 469.46
   dyad-hours and 939 participants, clearing both targets. But the
   naturalistic-only pool tops out near 470 dyad-hours, so a target above ~460
   requires improvised data or a second vendor — and V00 improvised-only *fails*
   the 500-participant floor at 485 while holding 2.1× the hours. Volume and
   diversity point in opposite directions and this is your and the PI's call.
3. **The acceptance-rate audit failure threshold.** The audit is specified,
   including the gesture-activity-decile cell that matters most, but this session
   sets no selection thresholds. You choose the factor at which a cell's
   acceptance rate counts as a failure.
4. **Whether a box-cropped Panel A variant is needed.** Portrait 2160×3840 clips
   give a 216-px-wide panel. Overlays are legible after the fix, but individual
   finger joints are small. Cropping would trade away the "participant leaves
   frame" context.
5. **Whether to store `joints_root_relative_mm` alongside `pose_axis_angle`.**
   Redundant (recoverable by FK) but saves re-running FK every epoch, at 22.45
   GiB for a 300-hour subset.
6. **Review and approve `docs/pipeline_spec_v1.md` before any implementation.**
   Nothing in it is built.

### 8.3 Now known to be expensive or impossible

| item | status |
|---|---|
| Detecting an implausible-but-present hand pose | **Not possible with released signals.** Availability is identical to box validity, and no exact-zero or frozen pose appears on valid spans. Needs a kinematic plausibility check or a pose prior — new work. |
| Detecting within-file audio/video desync | **Not possible with released signals.** Only container-level duration mismatch is available, and VAD never overran the media on any dev span. Needs a cross-modal probe. |
| An unconfounded jitter signal | **Not available as measured.** Residualization is arithmetically effective but unvalidated against human judgement. |
| Corpus-wide Stage 1 | **~3 node-hours**, not cheap but entirely affordable. Multi-node scaling beyond the ~170 MiB/s single-node ceiling is **unverified**. |
| Reading MP4/WAV in Stage 1 | **13.1× more bytes** for no additional Stage-1 signal. Confirmed avoidable. |
| Server-side quota at the clean output root | **Still unverified.** No quota client is available; `df` shows 74.417 TiB. |
| Establishing whether V00's 51 bare/suffixed ID pairs are one person | **Blocked.** Needs vendor documentation, or a face/voice identity check that would require its own privacy review. |
| Stage 4 FLAC transcode budget | **Unmeasured.** Must be benchmarked before scheduling; no estimate is offered. |
| Determining the upstream meaning of `smplh:is_valid` / `movement:is_valid` | **Still unverified after two sessions.** Needs vendor or paper-author input. M-3 added one clue (frozen hand pose on invalid spans). |
