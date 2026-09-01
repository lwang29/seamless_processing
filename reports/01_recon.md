# Session 1 — Seamless Interaction reconnaissance and minimal harness

Date: 2026-08-04 (America/Los_Angeles)  
Scope: `improvised/dev` and `naturalistic/dev` only  
Source: `./seamless_interaction` (read-only)  
Status: measurement only; no repairs, filtering thresholds, or selection decisions

## 1. Scope, safety, and populations

`./seamless_interaction` resolves to
`/simurgh2/datasets/seamless_interaction`. `findmnt` reports the source as an
NFSv4.2 mount with the `ro` option. No command in this session wrote below that
path.

Three deterministic manifests bound all analytical participant-payload reads:

| manifest | purpose | rows |
|---|---|---:|
| `configs/dev_sample.csv` | shared schema, validity, and harness sample | 120 |
| `configs/dyad_audit.csv` | 14 complete dyads for paired-stream checks | 28 |
| `configs/movement_audit.csv` | V00 supplement needed because movement flags are vendor-limited | 37 |
| **deduplicated union** | all participant-files touched by bounded analyses | **184** |

The union has 50 V00 files from each label, hence exactly 100 files containing
`movement:is_valid`. The harness itself uses only the original 120-row manifest.
The deterministic seeds and allocation logic are in
`scripts/make_dev_sample.py` and `configs/recon.yaml`.

The D1 throughput benchmark used 16 deterministic dev bundles, three of which
overlap this analytical union. Total unique participant bundles read anywhere
in the session are therefore **197**, below the approximately 200-file cap.

Private rendered media are under `artifacts/private_invalid_overlays/`, which is
git-ignored; the directory is mode `0700` and files are mode `0600`.

### Git caveat

Normal `git init` is impossible in this execution environment because `./.git`
is mounted as an empty read-only tmpfs. The functional repository is therefore
stored at `./.git-session` and used as:

```bash
git --git-dir=.git-session --work-tree=. <command>
```

This is a cluster/execution-platform constraint, not a property of the project.

## 2. D1 — Environment and resource report

### Runtime and current allocation

| item | observed value |
|---|---|
| host | `simurgh2.stanford.edu` |
| Slurm version | 25.11.5 |
| job / account / partition | `16499428` / `simurgh` / `simurgh-interactive` |
| allocation | 1 task, 16 CPUs, 200 GiB RAM, 2 GPUs |
| GPUs | 2 × NVIDIA L40S, 46,068 MiB each; driver 610.43.02 |
| cgroup CPUs | `8-15,72-79` |
| allocation end | 2026-08-07 10:59:43 PDT |
| account queue at observation | 134 running, 10 pending |

No additional Slurm job was submitted. All session processing used the approved
interactive allocation and no more than four concurrent workers.

### Slurm limits and quota-visible state

| setting | observed value |
|---|---|
| `simurgh` default / maximum time | 7 days / 21 days |
| `simurgh-interactive` default / maximum time | 6 hours / 3 days |
| `MaxArraySize` | 1001 (indexes through 1000) |
| `MaxJobCount` / `MaxStepCount` | 500000 / 40000 |
| user association group job/TRES limits | none reported by `sacctmgr` |
| `simurgh` QoS per-user GPU ceiling | 100 GPUs |

Exact server-side filesystem soft/hard quotas are **unverified**: `quota`,
`repquota`, `fs`, `lfs`, and comparable clients are unavailable. Slurm's
association output reports no explicit group job, submit, wall-time, or TRES
limits for the user/account.

### Filesystems and available capacity

| path | filesystem / mode | reported capacity |
|---|---|---|
| repository | NFSv4.2, read-write | 20 GiB total; 11 GiB available |
| source dataset | NFSv4.2, **read-only** | 402 TiB total; 85 TiB available |
| `/simurgh/group` | NFSv4.2, read-only in this context | 516 TiB total; 865 GiB available |

The approximately 11 GiB available on the repository export is not enough for a
large curated media subset. Whether `df` is exposing an export size or a
per-user quota is **unverified**.

### Python and Conda

Base Python is 3.13.13 and intentionally has none of NumPy, pandas, PyArrow,
PyYAML, OpenCV, pytest, or Torch. The existing read-only environment below is
usable without installation:

```bash
source /simurgh/group/lw29/miniconda3/etc/profile.d/conda.sh
conda activate ViBES
```

| package | version |
|---|---|
| Python | 3.10.20 |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |
| PyArrow | 24.0.0 |
| PyYAML | 6.0.3 |
| OpenCV | 4.11.0 |
| SciPy | 1.15.2 |
| SoundFile | 0.13.1 |
| pytest | 9.1.1 |
| Torch | 2.11.0+cu128 |

`conda run -n ViBES ...` fails with `NoWritableEnvsDirError`; activation or the
absolute interpreter works:

```bash
/simurgh/group/lw29/conda/envs/ViBES/bin/python
```

No package was installed into the shared environment.
`environment.yml` records all Session-1 runtime/test dependencies (including
pandas, OpenCV, and SciPy) for creation in a writable Conda location later; it
was not solved or installed during this session.

### Compute-node outbound network

DNS and HTTP HEAD requests from `simurgh2` succeeded for `pypi.org`,
`huggingface.co`, and `repo.anaconda.com` (HTTP 200). Thus this compute node
currently has outbound access. This is not guaranteed for future scheduled
nodes. No package, model, page body, or checkpoint was downloaded.

### FFmpeg / ffprobe

Both are `/usr/bin` version `6.1.1-3ubuntu5`. H.264, HEVC, AAC, and PCM decoding
are present. Reported acceleration APIs are VDPAU, CUDA, VAAPI, QSV, DRM,
OpenCL, and Vulkan; NVIDIA CUVID H.264/HEVC decoders are listed.

<details>
<summary>Exact FFmpeg build configuration</summary>

```text
--prefix=/usr --extra-version=3ubuntu5 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --disable-omx --enable-gnutls --enable-libaom --enable-libass --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libglslang --enable-libgme --enable-libgsm --enable-libharfbuzz --enable-libmp3lame --enable-libmysofa --enable-openjpeg --enable-libopenmpt --enable-libopus --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-openal --enable-opencl --enable-opengl --disable-sndio --enable-libvpl --disable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-ladspa --enable-libbluray --enable-libjack --enable-libpulse --enable-librabbitmq --enable-librist --enable-libsrt --enable-libssh --enable-libsvt-av1 --enable-libx264 --enable-libzmq --enable-libzvbi --enable-lv2 --enable-sdl2 --enable-libplacebo --enable-librav1e --enable-pocketsphinx --enable-librsvg --enable-libjxl --enable-shared
```

</details>

### Measured source-read throughput

These are raw reads to memory only: no parsing, decompression, decode, writes,
or cache dropping. Every bundle came from a dev split.

| trial | participant bundles | files | bytes | elapsed | aggregate throughput |
|---|---:|---:|---:|---:|---:|
| first pass, 1 worker | 8 | 32 | 649,025,442 | 23.7986 s | 26.008 MiB/s |
| immediate warm repeat | 8 | 32 | 649,025,442 | 0.2969 s | 2,085.042 MiB/s |
| disjoint first pass, 4 workers | 8 | 32 | 1,198,346,335 | 10.0324 s | 113.914 MiB/s |

The first passes are cache-uncontrolled, not provably cold. The approximately
80× warm-cache difference is large enough that later throughput reports must
state cache conditions and file selection.

## 3. D2 — Observed on-disk schema

### Actual layout

```text
seamless_interaction/
├── improvised/{dev,test,train}/
│   └── dev/<shard-group>/<shard-id>/<participant>.{json,mp4,npz,wav}
│       plus dev/<shard-group>/<shard-id>.tar
└── naturalistic/{dev,test,train}/
    └── dev/<shard-group>/<shard-id>/<participant>.{json,mp4,npz,wav}
```

This is not a flat set of modality directories. Improvised dev retains tar
archives alongside extracted payloads; naturalistic dev does not. Naturalistic
`0002` shard IDs have 12 gaps, so shard numbering is not safely enumerable by a
contiguous integer range.

| label | group | shard dirs | participant basenames | sibling tars |
|---|---:|---:|---:|---:|
| improvised | 0000 | 53 | 500 | 53 |
| improvised | 0001 | 63 | 184 | 63 |
| naturalistic | 0000 | 46 | 500 | 0 |
| naturalistic | 0001 | 42 | 500 | 0 |
| naturalistic | 0002 | 90 | 435 | 0 |

Every one of the 2,119 extracted dev basenames has `.json`, `.mp4`, `.npz`, and
`.wav` siblings. This only verifies path presence, not usable contents.

### Design-document §0 assumptions versus this staging

This table separates facts observed in the payload from claims that appear in
the paper/card-derived design document. “Unverified” is intentional: absence of
provenance metadata is not evidence that the paper's processing description is
false.

| §0 claim or working assumption | staging verdict | direct observation |
|---|---|---|
| Data are addressable through flat modality directories | **Refuted** | Payloads are participant bundles under two numeric shard levels; improvised also retains adjacent tar copies. |
| Transcript and VAD are separate JSONL modalities | **Refuted** | Both are arrays embedded in each participant JSON. No JSONL exists locally. |
| `filelist.csv`, `interactions.csv`, `participants.csv`, and `relationships.csv` are available for joins/availability checks | **Refuted for this local staging** | None exists below the staged root. Their official contents remain unverified. |
| Sibling-file presence establishes a usable multimodal record | **Refuted** | Two sampled V01 bundles have all four paths but zero annotation frames, no video stream, an empty WAV header, and empty JSON arrays. |
| `movement_v4:is_occluded` and `pred_vertices` are available on at least part of staged dev | **Refuted for the bounded union; full staging unverified** | Neither key occurs in any of 184 inspected NPZ headers. No full-corpus scan was performed. |
| Basic movement features/validity are generally available | **Refuted** | They occur only in the 100 bounded V00 files and are absent from all 84 bounded non-V00 files. |
| Released video is universally 1080p | **Refuted** | Native MP4 rasters include 1080x1920, 2160x2160, 2160x3840, and 3840x2160. |
| Released streams are exactly 30 Hz | **Refuted** | Both exact 30 and several approximately 29.97 rates occur, including variation within a vendor. |
| A canonical `beta=0` body shape is represented in the payload | **Unverified** | No beta, shape, vertex, 3D-joint, skeleton, or body-model asset is staged. Omission is compatible with—but does not prove—an implicit canonical template. |
| HMR/ViTPose/HaMeR provenance and the hand-to-wrist post-process can be recovered from each file | **Unverified** | Key shapes are compatible with the described SMPL-H parameters, but files contain no estimator/model versions or hand-root transform metadata. |
| Camera frame, metric units, and millimetre conversion are known from the arrays | **Unverified** | Translation is strongly image-coupled, but units, intrinsics, origin, and formal axis convention are absent. |
| The third keypoint value is a bounded confidence probability | **Refuted as bounded; semantics unverified** | It reaches 1.205525 and 2.68215% of bounded-sample values exceed one. |
| `numpy.load(..., mmap_mode="r")` avoids reading/decompressing members | **Refuted** | NPZ members use DEFLATE; member access decompresses them rather than true-mapping array bytes. |
| Both participant streams always share frame counts/time origins | **Partly refuted** | Annotation counts match in 14/14 dyads, but video counts and starts each match only 13/14; absolute WAV origins are unavailable. |
| Dev/test participant disjointness and documented corpus totals can be verified locally | **Unverified** | Required tables are absent, and this session intentionally did not scan train/test or the full corpus. |

### Counts and sizes (dev only)

| label | storage | modality | files | exact bytes | GiB |
|---|---|---|---:|---:|---:|
| improvised | extracted | JSON | 684 | 48,284,078 | 0.045 |
| improvised | extracted | MP4 | 684 | 95,283,955,267 | 88.740 |
| improvised | extracted | NPZ | 684 | 11,267,893,034 | 10.494 |
| improvised | extracted | WAV | 684 | 30,155,690,232 | 28.085 |
| improvised | archive copy | TAR | 116 | 136,761,538,560 | 127.369 |
| naturalistic | extracted | JSON | 1,435 | 72,605,586 | 0.068 |
| naturalistic | extracted | MP4 | 1,435 | 128,200,943,088 | 119.396 |
| naturalistic | extracted | NPZ | 1,435 | 18,797,193,091 | 17.506 |
| naturalistic | extracted | WAV | 1,435 | 50,784,609,310 | 47.297 |

Unique extracted dev payloads total 8,476 files / 334,611,173,686 bytes
(311.631 GiB). Including the retained improvised tar copies gives 8,592
physical files / 471,372,712,246 bytes (439.000 GiB). Two deterministic tar
spot checks found exact member names and sizes matching extracted siblings;
byte identity across all tar members is **unverified**.

### NPZ schema: exact 20-file inspection

The 20 files were selected reproducibly: sort each label's 60 manifest rows by
`file_id`, then draw 10 per label without replacement using one continuing
`random.Random(20260804)` instance.

| file ID | label | vendor | N | NPZ profile | MP4 stream | WAV duration (s) | JSON/MP4/NPZ/WAV paths |
|---|---|---|---:|---|---|---:|---|
| `V00_S0925_I00000487_P0816` | improvised | V00 | 5,160 | core + movement | yes | 172 | all present |
| `V00_S0700_I00000135_P0852` | improvised | V00 | 5,100 | core + movement | yes | 170 | all present |
| `V03_S0203_I00000538_P1437` | improvised | V03 | 5,934 | core | yes | 198 | all present |
| `V03_S0209_I00000578_P1555` | improvised | V03 | 16,364 | core | yes | 546 | all present |
| `V01_S0346_I00000726_P1694` | improvised | V01 | 0 | core, zero-length | **no** | unavailable | all present |
| `V00_S2025_I00001080_P1281A` | improvised | V00 | 7,020 | core + movement | yes | 234 | all present |
| `V00_S0644_I00000770_P0799` | improvised | V00 | 5,700 | core + movement | yes | 190 | all present |
| `V00_S0696_I00000135_P0847` | improvised | V00 | 3,660 | core + movement | yes | 122 | all present |
| `V03_S0251_I00000375_P1606` | improvised | V03 | 6,354 | core | yes | 212 | all present |
| `V00_S2021_I00001116_P1277A` | improvised | V00 | 7,620 | core + movement | yes | 254 | all present |
| `V00_S0515_I00000377_P0371` | naturalistic | V00 | 7,800 | core + movement | yes | 260 | all present |
| `V00_S0255_I00000134_P0350` | naturalistic | V00 | 6,300 | core + movement | yes | 210 | all present |
| `V03_S1605_I00000242_P4590` | naturalistic | V03 | 7,193 | core | yes | 240 | all present |
| `V02_S5054_I00000251_P5062` | naturalistic | V02 | 5,040 | core | yes | 168 | all present |
| `V02_S5054_I00000238_P5062` | naturalistic | V02 | 5,580 | core | yes | 186 | all present |
| `V00_S0152_I00000383_P0217` | naturalistic | V00 | 4,080 | core + movement | yes | 136 | all present |
| `V01_S1650_I00000132_P1514` | naturalistic | V01 | 8,452 | core | yes | 282 | all present |
| `V03_S0509_I00000203_P1659` | naturalistic | V03 | 60 | core | yes | 2 | all present |
| `V03_S1221_I00000304_P3778` | naturalistic | V03 | 4,676 | core | yes | 156 | all present |
| `V03_S1096_I00000001_P5628` | naturalistic | V03 | 7,432 | core | yes | 248 | all present |

Thus every sampled pathname-level modality is present, but one sampled bundle
has no usable video/audio/annotation content and movement availability follows
vendor rather than sibling paths. Official `filelist.csv` flags cannot be
compared because that table is absent.

All 20 have these nine core keys:

| key | exact shape | dtype |
|---|---|---|
| `boxes_and_keypoints:box` | `(N, 4)` | float32 |
| `boxes_and_keypoints:is_valid_box` | `(N,)` | bool |
| `boxes_and_keypoints:keypoints` | `(N, 133, 3)` | float32 |
| `smplh:body_pose` | `(N, 21, 3)` | float32 |
| `smplh:global_orient` | `(N, 3)` | float32 |
| `smplh:is_valid` | `(N,)` | bool |
| `smplh:left_hand_pose` | `(N, 15, 3)` | float32 |
| `smplh:right_hand_pose` | `(N, 15, 3)` | float32 |
| `smplh:translation` | `(N, 3)` | float32 |

Nine of 20 (all selected V00 files) additionally have this movement profile:

| key | exact shape | dtype |
|---|---|---|
| `movement:EmotionArousalToken` | `(N, 1)` | float32 |
| `movement:EmotionValenceToken` | `(N, 1)` | float32 |
| `movement:FAUToken` | `(N, 1)` | float32 |
| `movement:FAUValue` | `(N, 24)` | float32 |
| `movement:alignment_head_rotation` | `(N, 3)` | float32 |
| `movement:alignment_translation` | `(N, 2, 3)` | float32 |
| `movement:emotion_arousal` | `(N, 1)` | float32 |
| `movement:emotion_scores` | `(N, 8)` | float32 |
| `movement:emotion_valence` | `(N, 1)` | float32 |
| `movement:expression` | `(N, 128)` | float32 |
| `movement:frame_latent` | `(N, 64)` | float32 |
| `movement:gaze_encodings` | `(N, 2)` | float32 |
| `movement:head_encodings` | `(N, 3)` | float32 |
| `movement:hypernet_features` | `(N/15, 5120)` | float32 |
| `movement:is_valid` | `(N, 1)` | **float32** |

No inspected file contains `is_occluded`, `pred_vertices`, a `movement_v4:*`
key, mesh vertices, betas/shape, 3D joints, camera intrinsics, coordinate-frame
metadata, units, frame rate, or model-version provenance.

NPZ members are DEFLATE-compressed. `numpy.load(..., mmap_mode="r")` does not
true-memory-map these arrays; member access decompresses data. The design
document's proposed mmap shortcut is therefore refuted for this staging.

One inspected bundle, `V01_S0346_I00000726_P1694`, has a valid 1,981-byte NPZ
whose nine core arrays all have `N=0`; its MP4 is a 261-byte, zero-stream
container; WAV is a 58-byte header; and JSON contains empty transcript/VAD
lists. A second union file, `V01_S0346_I00000727_P1693`, has the same empty
bundle pattern. Presence flags alone would incorrectly call both complete.

### Official metadata tables and availability flags

After pruning train/test paths, no `filelist.csv`, `interactions.csv`,
`participants.csv`, `relationships.csv`, other CSV, JSONL, `transcript/`, or
`vad/` directory exists in the local staging.

Therefore the requested official columns, dtypes, null rates, activity/prompt
joins, participant/relationship joins, and filelist availability-flag agreement
are **unverified because the files are unavailable**, not inferred from the
README. `configs/dev_sample.csv` is this project's generated manifest, not an
official substitute.

For the 120-row project manifest, all four sibling paths exist and none is zero
bytes. Logical content is different: movement is absent outside V00, and one
selected bundle has empty core arrays/streams. The correct availability model
must distinguish path presence, container/key presence, and nonempty content.

### Actual JSON schema

Transcript and VAD are arrays embedded in each participant `.json`, not JSONL:

```text
id: string
metadata:transcript: list[TranscriptRecord]
metadata:vad: list[VADRecord]
```

```text
TranscriptRecord = {
  words: list[{word: string, start: number|null, end: number|null, score?: number}],
  start: number,
  end: number,
  transcript: string
}
VADRecord = {start: number, end: number}
```

Across the 20 files there are 570 transcript segments, 5,550 word records, and
822 VAD records. Twelve word records have `start: null`, `end: null`, and omit
`score`. One file adds five `annotations:*` top-level arrays with records
`{annotation, start_ts, end_ts}`; consumers must tolerate additional keys.

Two transcript records, copied verbatim from the pretty-printed participant
JSON:

```json
{
            "words": [
                {
                    "word": "Right.",
                    "start": 69.668,
                    "end": 69.928,
                    "score": 0.504
                }
            ],
            "start": 69.668,
            "end": 69.928,
            "transcript": "Right."
        }
```

```json
{
            "words": [
                {
                    "word": "Yeah.",
                    "start": 516.452,
                    "end": 520.415,
                    "score": 0.822
                }
            ],
            "start": 516.452,
            "end": 520.415,
            "transcript": "Yeah."
        }
```

Two VAD records, copied verbatim:

```json
{
            "start": 258.77,
            "end": 259.182
        }
```

```json
{
            "start": 199.49,
            "end": 200.798
        }
```

## 4. D3 — `is_valid` characterization

### Population and mask availability

The deduplicated audit union contains 184 files (88 improvised, 96
naturalistic). Two V01 bundles have zero frames, leaving 182 usable files for
SMPL-H/box analysis. There are no mask-length mismatches. Machine-readable
tables are under `outputs/recon/validity_audit/` (git-ignored generated output),
produced by `scripts/analyze_validity.py`.

| mask | on-disk schema | present | usable | availability finding |
|---|---|---:|---:|---|
| `smplh:is_valid` | `(T,) bool` | 184/184 | 182 | all vendors |
| `boxes_and_keypoints:is_valid_box` | `(T,) bool` | 184/184 | 182 | all vendors |
| `movement:is_valid` | `(T,1) float32`, values `{0,1}` | 100/184 | 100 | V00 only; absent from all 84 bounded non-V00 files |

Missing movement is recorded as unknown/unavailable, never treated as valid.

### Invalid fraction across files

Percentages below are distributions of the per-file invalid fraction. “Pooled”
weights by frames and is included separately so long files do not silently
replace the requested across-file distribution.

| mask | files | files with any invalid | q0 | q25 | q50 | q75 | q90 | q95 | q99 | max | pooled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SMPL-H | 182 | 114 | 0 | 0 | 0.3384 | 1.9023 | 7.5132 | 29.5500 | 81.3639 | 100 | 4.4708 |
| movement | 100 | 67 | 0 | 0 | 2.7864 | 19.8153 | 57.0979 | 62.1564 | 100 | 100 | 14.7031 |
| box | 182 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0348 | 17.3432 | 0.09343 |

Counts behind the pooled rates are 50,773 / 1,135,666 SMPL-H frames,
93,600 / 636,600 movement frames, and 1,061 / 1,135,666 box frames.

Label-level pooled SMPL-H invalidity is 5.0274% improvised versus 3.9070%
naturalistic; movement is 15.3386% versus 14.0578%. This is characterization,
not a threshold or quality ranking.

Notable files:

- `V01_S0172_I00001232_P1316` is nonempty but SMPL-H-invalid for all 1,800
  frames.
- Two improvised movement files are 100% invalid (5,520 and 6,780 frames).
- Naturalistic has zero box-invalid frames in this bounded union.

### Invalid-run structure

| mask | runs | invalid frames | q25 | median | q75 | q90 | q95 | q99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SMPL-H | 2,019 | 50,773 | 1 | 3 | 9 | 31 | 55.1 | 402.5 | 3,740 |
| movement | 450 | 93,600 | 60 | 120 | 180 | 360 | 600 | 1,440 | 6,780 |
| box | 3 | 1,061 | — | 42 | — | — | — | — | 1,009 |

SMPL-H has 700 isolated one-frame runs (34.67% of runs), but they account for
only 1.379% of invalid frames. Twenty-nine runs of at least 300 frames account
for 56.434% of all SMPL-H-invalid frames. Invalidity is therefore mostly bursty
by frame mass even though isolated events are common by run count.

Every movement run starts on a frame divisible by 60, and every run length is a
multiple of 60 (`gcd=60`). There are no isolated movement failures. This exact
two-second block structure, plus the encoding results below, looks like
blockwise missing/failed feature production rather than ordinary framewise pose
confidence.

The three box runs have lengths 10, 42, and 1,009. The 1,009-frame run carries
95.099% of all box-invalid frames.

### Pairwise agreement

Invalid is the positive class. Agreement is reported but interpreted cautiously
because both-valid frames dominate it.

| pair / overlap | both valid | first-only invalid | second-only invalid | both invalid | agreement | invalid Jaccard | phi |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMPL-H / movement; 100 V00 files, 636,600 frames | 538,575 | 4,425 | 90,502 | 3,098 | 85.0884% | 3.1604% | 0.08176 |
| SMPL-H / box; 182 files, 1,135,666 frames | 1,084,893 | 49,712 | 0 | 1,061 | 95.6227% | 2.0897% | 0.14136 |
| movement / box; 100 V00 files, 636,600 frames | 543,000 | 93,600 | 0 | 0 | 85.297% | 0 | undefined/degenerate |

Every observed box-invalid frame is SMPL-H-invalid, but only 2.0897% of
SMPL-H-invalid frames are box-invalid. Movement and SMPL-H are also largely
different signals: only 3.3098% of movement-invalid frames are SMPL-H-invalid,
and their across-file invalid-fraction correlation is 0.2183. All box failures
occurred in non-V00 files, where movement does not exist, so the zero overlap in
the last row must not be generalized.

### What arrays contain on invalid frames

No NaN or infinity occurs in any aligned array, on either valid or invalid
frames. Missingness is encoded in three materially different ways:

| family | observed invalid-frame contents |
|---|---|
| box/keypoints | All 1,061 invalid frames have exactly all-zero box and 133×3 keypoints. Within the three runs, 99.7172% equal the previous frame because zero is repeated; none equals the most recent valid frame. |
| movement | Eleven of 13 frame-aligned arrays are exactly all-zero on 100% of 93,600 invalid frames. `FAUToken`/`FAUValue` are zero on 99.9498%; their 47 nonzero cases are at run boundaries. Typical equal-previous rate is 99.5320%, reflecting repeated zeros, not last-valid hold. `hypernet_features` is `(T/15,5120)` and was correctly excluded from frame-aligned analysis. |
| SMPL-H | Invalid pose/translation remains finite, nonzero, and usually varying. Among 40,047 invalid frames with a preceding valid frame, none of the complete SMPL-H arrays exactly equals that last valid frame. Invalid values are neither NaN nor zero-fill nor a simple last-valid hold. |

SMPL-H exact current-equals-previous rates on invalid frames are 1.7809% for
body pose/global orientation, 27.1843% for the left hand, 33.1495% for the right
hand, and 2.0843% for translation. The elevated hand rates mean freezes occur,
but not as a universal encoding.

Only 49.0474% of invalid body-pose frames, 91.7567% of global orientation,
68.7277% of left hand, 61.4400% of right hand, and 86.9765% of translation lie
entirely within that file's per-feature valid minima/maxima. These are numerical
range checks, not anatomical plausibility checks. The fully-invalid 1,800-frame
file contains finite, nonzero, varying pose and translation: plausible-looking
values with no valid comparator.

On every box-invalid frame, SMPL-H translation is a huge finite sentinel rather
than zero/NaN. Observed examples are approximately
`[-2.16e12,-2.16e12,8.44e13]` and
`[-2.16e12,-3.84e12,1.50e14]`; valid translation components across the audit
remain within `[-0.719,0.777]`, `[-0.0166,0.860]`, and `[20.959,63.439]`.
Exactly the box-invalid frames—and no others—have `|translation| > 1000` in
this audit. This is an observed sentinel characterization, not a selected gate.

### Ten private visual checks

The renderer uses OpenCV to draw released COCO-WholeBody 2D keypoints, box, and
all three mask states, then pipes raw frames to FFmpeg/libx264. Clips and
boundary-centered contact sheets are under
`artifacts/private_invalid_overlays/` and are mode `0600`.

| clip | target | visual observation |
|---|---|---|
| `invalid_01.mp4` | 42-frame box run | The participant rapidly exits the right edge with motion blur. Box invalidity begins when only a sliver remains, continues over an empty scene, and persists briefly during re-entry. |
| `invalid_02.mp4` | 10-frame terminal box run | Video cuts to black. The first black frame is still flagged valid and displays stale-looking keypoints; the next frame turns box/SMPL-H invalid and arrays become zero/sentinel. This reveals a one-frame optimistic edge case. |
| `invalid_03.mp4` | isolated SMPL-H frame after a long leading run | Person is fully visible; 2D body/hands remain plausible across the transition. No video-level cause is apparent. |
| `invalid_04.mp4` | start of 849-frame SMPL-H run | Person remains fully visible and nearly static. Strong side glare and dark clothing are present, but there is no visible discontinuity at the flag transition; causality is unverified. |
| `invalid_05.mp4` | isolated SMPL-H frame | The participant makes a wide gesture and one hand/forearm reaches the frame boundary. 2D body tracking remains plausible; truncation is a plausible contributor, not proven. |
| `invalid_06.mp4` | start of 2,464-frame SMPL-H run | Person is fully in frame; a cap shadows part of the face and the hands are low. No disappearance, partner occlusion, or 2D tracking collapse occurs at the transition. |
| `invalid_07.mp4` | 60-frame movement run | Fully visible active gesture with one hand near the chest/face. The two-second movement-invalid block starts without a visual discontinuity; SMPL-H and 2D stay valid. |
| `invalid_08.mp4` | start of 1,200-frame movement run | Fully visible; the participant turns/lowers the head. Movement validity switches at an exact block boundary with no loss of 2D tracking. |
| `invalid_09.mp4` | 60-frame movement run | Fully visible, leaning forward with hands clasped. No visible change at the two-second invalid boundary. |
| `invalid_10.mp4` | start of 2,760-frame movement run | Fully visible and gesturing. Movement briefly toggles valid then returns invalid while the image and 2D tracking remain continuous. |

Visual conclusion: box invalidity has clear recording/visibility meaning in the
observed examples (out of frame or black terminal video). SMPL-H invalidity is
broader: it sometimes coincides with truncation but often occurs with a fully
visible participant and stable 2D keypoints. Movement invalidity is not a pose
visibility flag in these examples; together with the 60-frame block structure
and zero-fill, it marks unavailable/failed movement-feature blocks. The three
masks must not be treated as interchangeable votes for the same event.

## 5. D4 — Assumption audit

The NPZ-key inventory and media-header audit cover the bounded 184-file union.
Payload-heavy pose, translation, keypoint, and box measurements use the
120-file main manifest (118 nonempty files). Pose summaries below use 716,216
`smplh:is_valid` frames and 714,760 consecutive-valid frame pairs. Paired-file
results use all 14 dyads / 28 participant-files in `configs/dyad_audit.csv`.
The exact bounded definitions and raw summaries are reproducible with
`scripts/audit_assumptions.py`, which atomically writes the git-ignored
`outputs/recon/assumption_audit.json`.

| design-document assumption or question | verdict | observed evidence |
|---|---|---|
| A canonical body shape with $\beta=0$ is actually present | **Unverifiable** | No `beta` or shape key occurs in any of the 184 bounded NPZs. Omission is compatible with an implicit canonical shape, but does not prove one. |
| Inter-joint distances are constant within and across participants | **Unverifiable** | The release contains no 3D joints, vertices, skeleton offsets, or body-model asset from which to measure distances. |
| `smplh:translation` has metric units that can be converted to millimetres | **Unverifiable** | There is no unit, camera calibration, or scale metadata. Raw coordinate 2 spans 26.579--63.350 while coordinates 0/1 remain mostly below one. |
| Translation/global orientation have a known origin and axis convention | **Unverifiable** | Translation is strongly image/camera-coupled, but the exact physical origin, up/forward axes, and rotation-frame convention are not encoded. |
| The coordinate frame is camera-relative rather than world-ish | **Unverifiable, with strong camera-relative evidence** | `tx` follows horizontal box position, `ty` follows vertical box position, and `tz` is depth-like. These relationships do not substitute for missing intrinsics/extrinsics or formal frame metadata. |
| Body, hand, and global rotations are axis-angle rather than matrices or 6D | **Verified among those alternatives** | Stored suffixes are `3`, `21x3`, and `15x3`; values exceed $\pm1$, excluding stored matrix elements, and there is no six-value dimension. The vectors are not canonicalized to norm $\le\pi$. |
| Released hand poses can be shown to be transformed into each wrist frame | **Unverifiable** | Fifteen three-vectors per hand are consistent with SMPL-H local finger rotations, but no hand-root transform or processing metadata exposes the claimed HaMeR-to-wrist conversion. |
| Every released stream is exactly 30 fps | **Refuted** | The bounded union contains `30/1`, `30000/1001`, and `45000/1501` stream rates, plus two zero-stream placeholders. Rate also varies within a vendor by label/resolution. |
| `abs(n_frames/30 - audio_duration)` is an unbiased duration-consistency measure | **Refuted** | Its maximum is 0.600 s on the bounded union, versus 0.0333 s when video frame count is divided by measured average fps. The nominal-30 value accumulates expected drift on 29.97-fps media. |
| Both participant files always have equal frame counts and aligned time origins | **Partly refuted** | NPZ frame counts agree in 14/14 dyads, but video counts and video starts each agree in only 13/14. Absolute WAV origins are unavailable. |
| Keypoints are normalized, universally in a released 1080p frame, or universally in original 4K portrait space | **Refuted** | Coordinates are pixel-valued and track each file's actual MP4 raster: 1080x1920, 2160x2160, 2160x3840, or 3840x2160. |
| The third keypoint channel is a bounded confidence probability | **Unverifiable semantics; refuted as a bounded probability** | A third float channel exists, but spans 0--1.205525 and 2.682% of values exceed one. |
| Boxes are `[xmin, ymin, xmax, ymax]` | **Verified** | On all 745,844 valid-box frames in the main sample, coordinate 2 exceeds coordinate 0, coordinate 3 exceeds coordinate 1, and all endpoints lie in the native raster. |

### Shape and staged-asset limits

All 184 NPZ headers have one of two key sets: 100 V00 files have the 24-key
schema including `movement:*`, while 84 files have only the nine core keys.
Across both sets, zero key names contain `beta`, `shape`, `joint`, or `vert`.
No SMPL/SMPL-H body-model asset is present in the repository. The relevant
stored layouts are:

| array | layout | dtype |
|---|---|---|
| `smplh:translation` | `(N, 3)` | float32 |
| `smplh:global_orient` | `(N, 3)` | float32 |
| `smplh:body_pose` | `(N, 21, 3)` | float32 |
| `smplh:left_hand_pose` | `(N, 15, 3)` | float32 |
| `smplh:right_hand_pose` | `(N, 15, 3)` | float32 |

Consequently, an external body model plus an explicit choice of template,
shape, joint convention, and units would be assumptions introduced by this
project rather than properties observed in the staged payloads.

### Translation and coordinate evidence

Valid-frame raw translations in the main sample are:

| coordinate | minimum | p1 | median | p99 | maximum |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.718989 | -0.322129 | -0.003885 | 0.162418 | 0.587157 |
| 1 | -0.016611 | 0.165669 | 0.309234 | 0.826741 | 0.860094 |
| 2 | 26.578753 | 31.045645 | 39.162170 | 55.466078 | 63.349545 |

The consecutive-valid translation-delta norm has p50 0.051539, p95 0.278940,
p99 0.491504, and maximum 5.782309, all in **unverified raw units per frame**.
Median within-file coordinate ranges are 0.143548, 0.083087, and 4.548368 for
coordinates 0--2 respectively.

The following comparisons use native-raster-normalized boxes and every tenth
valid frame:

| observed relationship | pooled-frame Pearson $r$ | across-file-mean Pearson $r$ |
|---|---:|---:|
| translation 0 vs. box center x / width | 0.8755 | 0.9317 |
| translation 1 vs. box center y / height | 0.9510 | 0.9405 |
| translation 2 vs. box height / height | -0.6837 | -0.6164 |
| translation 2 vs. inverse normalized box height | 0.6424 | 0.5554 |

A projection-like regression of `translation[0] / translation[2]` on normalized
box-center x has $R^2=0.8032$ and crosses zero at 0.4993 image widths. This is
data evidence for positive-x=image-right, positive-y=image-down, and a
distance-like positive third coordinate. It does **not** verify physical units,
camera intrinsics, the precise origin, or whether the published authors called
this frame camera or world coordinates. `global_orient` component medians are
`[3.1152, -0.0140, 0.00062]`; the dominant near-$\pi$ first component is
observed, but its physical axis meaning is unverified.

### Rotation representation and wrap discontinuities

Valid-frame numerical ranges are:

| array | component min / max | vector-norm p99 / max | raw adjacent-vector delta p99 / max |
|---|---:|---:|---:|
| global orientation | -2.328 / 3.720 | 3.459 / 3.851 | 0.0249 / 6.298 |
| body pose | -2.980 / 4.711 | 4.386 / 4.712 | 0.0999 / 7.462 |
| left hand pose | -2.656 / 4.711 | 1.495 / 4.711 | 0.1203 / 6.293 |
| right hand pose | -2.625 / 4.532 | 1.502 / 4.661 | 0.1208 / 6.299 |

Direct axis-angle subtraction creates representational spikes. The two global
orientation deltas above 3 radians have raw magnitudes 6.270--6.298 radians but
rotation-composition geodesic magnitudes only 0.0518--0.0536 radians. Of 3,686
left-hand raw deltas above 3 radians, 78.8% have geodesic magnitude below 0.1
radian. Later angular-velocity measurements therefore need rotation
composition/geodesic distance, not direct vector subtraction; this is a
representation requirement, not a filtering threshold.

### Native video rates and duration agreement

`r_frame_rate` and resolution in the bounded union are:

| label | vendor | raster | files | exact `r_frame_rate` |
|---|---|---|---:|---:|
| improvised | V00 | 1080x1920 | 50 | `30/1` |
| improvised | V01 | 1080x1920 | 3 | `45000/1501` = 29.980013 |
| improvised | V01 | 2160x2160 | 10 | `30/1` |
| improvised | V01 | no decodable stream | 2 | unavailable |
| improvised | V03 | 2160x3840 | 23 | `30000/1001` = 29.970030 |
| naturalistic | V00 | 1080x1920 | 50 | `30/1` |
| naturalistic | V01 | 1080x1920 | 14 | `30000/1001` |
| naturalistic | V01 | 2160x2160 | 4 | `30000/1001` |
| naturalistic | V02 | 1080x1920 | 8 | `30/1` |
| naturalistic | V03 | 2160x3840 | 14 | `30000/1001` |
| naturalistic | V03 | 3840x2160 | 6 | `30/1` |

Average frame rates, which incorporate actual stream frame counts and timing,
are exactly 30 for 120 files, exactly `30000/1001` for 53, and exactly
`45000/1501` for three. Four V01 square files average 29.993103 (one),
29.995049 (two), or 29.996815 (one); two V03 portrait files average 29.966719;
two files have no stream. The latter two bundles are nonzero-size but empty
placeholders: 261-byte MP4, 58-byte WAV, 1,981-byte zero-length-array NPZ.

| label/vendor/raster | files with duration | `abs(N_smplh/30 - wav)` median / max (s) | `abs(N_video/avg_fps - wav)` median / max (s) |
|---|---:|---:|---:|
| improvised V00, 1080x1920 | 50 | 0 / 0 | 0 / 0 |
| improvised V01, 1080x1920 | 3 | 0.133333 / 0.166667 | 0.006733 / 0.009244 |
| improvised V01, 2160x2160 | 10 | 0 / 0.200000 | 0.016667 / 0.033333 |
| improvised V03, 2160x3840 | 23 | 0.200000 / 0.600000 | 0.023667 / 0.032333 |
| naturalistic V00, 1080x1920 | 50 | 0 / 0 | 0 / 0 |
| naturalistic V01, 1080x1920 | 14 | 0.150000 / 0.266667 | 0.022050 / 0.031567 |
| naturalistic V01, 2160x2160 | 4 | 0.183333 / 0.200000 | 0.013800 / 0.025167 |
| naturalistic V02, 1080x1920 | 8 | 0 / 0 | 0 / 0 |
| naturalistic V03, 2160x3840 | 14 | 0.216667 / 0.366667 | 0.011700 / 0.033167 |
| naturalistic V03, 3840x2160 | 6 | 0 / 0 | 0 / 0 |

Across all 182 nonempty bundles, the nominal-30 mismatch has median 0 and
maximum 0.600 s; the actual-fps mismatch has median 0 and maximum 0.033333 s.
Union video-minus-NPZ frame-count deltas are `-3: 1`, `-2: 3`, `-1: 15`,
`0: 155`, `+1: 7`, `+5: 1`, plus two unprobeable files.

### Paired participant timing

| paired check | equal dyads |
|---|---:|
| SMPL-H frame count | 14/14 |
| box/keypoint frame count | 14/14 |
| movement frame count where present | 4/4 |
| WAV duration | 14/14 |
| embedded-audio stream start | 14/14 |
| video frame count | 13/14 |
| video stream start | 13/14 |
| video format duration | 13/14 |

The counterexample is interaction `V03_S0251_I00000140`: P1606 has NPZ/video
counts 10,789/10,789 and video start 0.033 s; P1607 has 10,789/10,790 and video
start 0 s. Both WAVs are exactly 360 s, and video format durations differ by
0.000366 s. In addition, both participants in `V01_S0346_I00000720` have videos
two frames shorter than their NPZs, and both in `V03_S0871_I00000117` have
videos one frame shorter. Only V00 carries a video timecode, always
`00:00:00:00`; other vendors lack it, and WAV probes expose no absolute start
timestamp. Equal WAV durations therefore do not verify a shared absolute time
origin.

### Keypoint and box coordinate semantics

The main sample contains 99,198,582 keypoints over 745,854 frames in 118
nonempty files. Coordinates scale with each probed native MP4 raster rather than
with one common 1080p or normalized frame.

| off-frame measurement | result |
|---|---:|
| aggregate point fraction | 0.86775% |
| fraction among third-channel-positive points | 0.86776% |
| per-file median | 0.0586% |
| per-file p95 | 4.408% |
| per-file maximum | 5.923% |

Per-file normalized extrema reach `x/W=-0.116`, `y/H=-0.0645`, `x/W=1.178`,
and `y/H=1.088`; positive third-channel values do not imply in-frame
coordinates. The third channel is finite everywhere, ranges 0--1.205525, has a
deterministic-sample median 0.945625 and p99 1.008232, is exactly zero for
0.00134% of points, and exceeds one for 2.68215%. Its presence is verified, but
calling it a calibrated confidence is not justified by the payload alone.

For boxes, all 745,844 valid-box frames have ordered `[xmin,ymin,xmax,ymax]`
coordinates whose endpoints lie within the native raster. Only 4.684% would
satisfy the corresponding `x+width <= W` and `y+height <= H` conditions under
an `[x,y,width,height]` interpretation, confirming the former representation.

### Consequence for the minimal harness

With current staged assets, `accel_mm_per_frame2` and metric 3D wrist speed are
**not defensible measurements**. Rotation parameters alone do not supply joint
positions or skeleton offsets, and the translation unit is unverified. Safe
Session-1 outputs can use raw-unit translation differences or geodesic angular
differences, explicitly named as such. Metric FK requires separately staged
SMPL-H assets and an explicit, validated choice of template, beta, units, and
joint convention; none should be silently inferred from the key names.

## 6. D5 — Minimal extraction harness

### Components and execution model

| component | path / behavior |
|---|---|
| configuration | `configs/harness.yaml`; source, manifest, output root, 4 s duration, 1 s hop, and 30 Hz window grid are data, not literals in the worker |
| CLI | `python -m seamless_curation --config ... [--limit N] [--index I] [--force]` |
| worker | `src/seamless_curation/harness.py`; one NPZ load, one WAV-header read, and one video-rate `ffprobe` per participant-file, followed by all full windows |
| outputs | one typed Parquet per participant-file plus one JSON completion marker; errors are non-completion JSON records and are retried |
| verification | `scripts/summarize_harness.py` asserts expected file/marker counts, no error/temp files, row totals, and provenance consistency |
| Slurm | `slurm/extract_dev_array.sbatch`; array `0-119%16`, one CPU, 8 GiB, 30 min, no GPU; **not submitted** |

Parquet and JSON writers create a UUID-named temporary file in the destination
directory, flush/fsync it, then call `os.replace`. The Parquet rename completes
before the completion-marker rename. A matching rerun skips only when output
and marker both exist and config hash, Git SHA, and clean-worktree status match.
Dirty runs are stamped `git_dirty=true` but deliberately cannot reuse markers,
because a Git SHA cannot identify uncommitted code. Configured output paths are
resolved and rejected if they escape the repository output root or enter the
read-only source.

### Four measurement columns

| column | exact Session-1 definition | limitation/status behavior |
|---|---|---|
| `valid_frac_all` | mean of the logical AND of SMPL-H, movement, and box masks in the window | `NaN` with `missing_masks:movement` if any mask is absent; missing never becomes valid |
| `accel_mm_per_frame2` | mean norm of the second difference of released root translation, over finite SMPL-H-valid triplets, multiplied by configured 1000 | **Provisional compatibility column, not verified millimetres and not joint/FK acceleration**; every row says `physical_units_verified=false`; `NaN` if no valid triplet |
| `wrist_speed_p90` | p90 of adjacent displacement for released COCO-WholeBody body-wrist indices 9/10 | native-raster pixels/frame; adjacent box-valid finite XY only; no confidence threshold |
| `duration_mismatch_s` | `abs(N_annotation / video.avg_frame_rate - WAV_header_duration)` | file-level value repeated on its windows; explicit config-rate fallback if a video stream exists but rate parsing fails; unavailable/zero-stream becomes `NaN` |

The fixed 120-frame/30-frame window grid remains nominal 30 Hz, independently
of duration measurement. No resampling or timing repair occurs. The requested
metric acceleration cannot be justified from the staged files (D4); retaining
the requested column name with explicit provisional basis/status proves the
plumbing without silently converting an unknown unit.

### Full 120-file verification

The clean code-hardening run was stamped with Git SHA
`1e45f8155c326b29f26abec3538491147a39e1c0` and config SHA-256
`d3e5b4413cd6aa28dbad81f6076f0747f8c7b9aba545eb565bf5f1c81c119f81`.
It completed in 29.19 s with 158,180 KiB maximum RSS on a warm source cache;
an earlier first full pass took 50.21 s. The immediate marker-only rerun took
0.28 s and skipped all 120 files.

| verification item | observed result |
|---|---:|
| participant Parquets / completion markers | 120 / 120 |
| window rows | 24,472 |
| rows by label | 13,763 improvised; 10,709 naturalistic |
| error records / leftover temporary files | 0 / 0 |
| full window lengths | 120 frames only |
| finite `valid_frac_all` / missing-movement `NaN` | 11,764 / 12,708 |
| finite acceleration / no-valid-triplet `NaN` | 23,913 / 559 |
| finite wrist speed / duration mismatch | 24,472 / 24,472 |

Four successfully processed files have zero window rows: the two N=0 V01
placeholder bundles and both participants in a 60-frame naturalistic V03
interaction. Their completion markers preserve `n_frames`, empty/short status,
keys, media-rate status, and provenance; “complete” means extraction completed,
not that the source is suitable for training.

## 7. D6 — Synthetic corruption tests

Three 180-frame V00 dev spans are named by both manifest index and expected
`file_id` in `tests/fixtures/dev_span.yaml`. All three masks are valid throughout,
and six boundary/interior frames per span were visually reviewed using private
2D-overlay contact sheets before marking them clean. Tests fail if manifest
reordering makes an index resolve to a different file.

| injected fault | exact injection | asserted response | asserted non-response / known coupling |
|---|---|---|---|
| frame drop / duplication | delete one middle sample; separately insert three exact copies | duration mismatch becomes exactly `1/30` s or `3/30` s; duplicate maximum exact-zero-velocity run increases | all-three validity fraction is unchanged; root acceleration is explicitly asserted to change because temporal deletion/duplication necessarily perturbs local second differences |
| known jitter | add alternating `+/-0.01` raw model units to translation coordinate 0 | provisional root-acceleration measurement increases | validity, 2D wrist speed, and duration mismatch are unchanged |
| blank hand | set released left-hand keypoint XY block (21 points) to exactly zero for 20 frames | scaffold `hand_availability_frac` falls by exactly `20/180` | the four harness measurements remain unchanged, including released validity masks; this deliberately demonstrates that a hand-availability detector is needed beyond body-wrist speed |

The drop/duplicate case cannot honestly satisfy strict detector orthogonality:
changing temporal samples also changes an acceleration statistic. The test
encodes this expected cross-response rather than labeling it a false positive.
A fourth semantic check asserts that an absent movement mask yields unknown
(`NaN`), never an all-valid window. Restart/atomic-output, dirty-marker,
path-escape, empty-array, and provenance behavior are also covered.

Final local result: **14 tests passed**, pytest-runnable in well under one
minute (4.32 s on the timed allocated-node run).

## 8. Reproduction commands

Run from the repository root. The absolute ViBES interpreter avoids the
read-only Conda registry issue.

### Safety, allocation, filesystem, and tools

```bash
readlink -f ./seamless_interaction
findmnt -T ./seamless_interaction -o TARGET,SOURCE,FSTYPE,OPTIONS
findmnt -T . -o TARGET,SOURCE,FSTYPE,OPTIONS
df -hT . ./seamless_interaction /simurgh/group/lw29

sinfo --version
sinfo -h -o '%P|%a|%l|%D|%c|%m|%G'
squeue -u lw29 -o '%.18i %.18P %.30j %.8u %.2t %.10M %.10l %.6D %R'
squeue -h -A simurgh -o '%T' | sort | uniq -c
scontrol show job 16499428
scontrol show partition simurgh --oneliner
scontrol show partition simurgh-interactive --oneliner
scontrol show config | rg '^(MaxArraySize|MaxJobCount|MaxStepCount)'
sacctmgr -P show assoc where user=lw29 format=Cluster,Account,User,Partition,QOS,DefaultQOS,GrpTRES,GrpJobs,GrpSubmit,MaxJobs,MaxSubmit,MaxWall,MaxTRES
sacctmgr -P show qos where name=normal,simurgh format=Name,Flags,GraceTime,GrpTRES,GrpJobs,GrpSubmit,MaxJobsPerUser,MaxSubmitJobsPerUser,MaxWall,MaxTRESPerUser

command -v python python3 conda ffmpeg ffprobe
/simurgh/group/lw29/conda/envs/ViBES/bin/python --version
ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -hwaccels
ffmpeg -hide_banner -decoders 2>/dev/null | rg '(^| )h264|(^| )hevc|(^| )aac|pcm_s16le'
```

Quota clients were checked with
`command -v quota repquota xfs_quota zfs fs lfs mmlsquota`; none was available.

### Outbound-network header-only probes

```bash
getent ahostsv4 pypi.org | head -n 1
getent ahostsv4 huggingface.co | head -n 1
getent ahostsv4 repo.anaconda.com | head -n 1
curl -sS -I --max-time 15 https://pypi.org/simple/ | head -n 8
curl -sS -I --max-time 15 https://huggingface.co/ | head -n 8
curl -sS -I --max-time 15 https://repo.anaconda.com/pkgs/main/linux-64/repodata.json | head -n 8
```

### Exact source-read benchmarks

Sequential first pass (repeat unchanged for the reported warm pass):

```bash
/simurgh/group/lw29/conda/envs/ViBES/bin/python -c $'import glob, os, time\nroots=["./seamless_interaction/improvised/dev/0000/0000","./seamless_interaction/naturalistic/dev/0000/0000"]\nnpzs=sum([sorted(glob.glob(root+"/*.npz"))[:4] for root in roots], [])\npaths=[]\nfor npz in npzs:\n    stem=os.path.splitext(npz)[0]\n    paths.extend(p for ext in (".npz",".wav",".mp4",".json") if os.path.isfile(p:=stem+ext))\ndef consume(path):\n    total=0\n    with open(path,"rb",buffering=0) as f:\n        while chunk:=f.read(8*1024*1024): total+=len(chunk)\n    return total\nt0=time.perf_counter(); total=sum(consume(p) for p in paths); dt=time.perf_counter()-t0\nprint("participants",len(npzs),"files",len(paths),"bytes",total,"seconds",f"{dt:.6f}","MiB_per_s",f"{total/dt/2**20:.3f}")'
```

Disjoint four-worker first pass:

```bash
/simurgh/group/lw29/conda/envs/ViBES/bin/python -c $'import glob, os, time, concurrent.futures as cf\nroots=["./seamless_interaction/improvised/dev/0000/0052","./seamless_interaction/naturalistic/dev/0002/0101"]\nnpzs=sum([sorted(glob.glob(root+"/*.npz"))[:8] for root in roots], [])\nbundles=[]\nfor npz in npzs:\n    stem=os.path.splitext(npz)[0]\n    bundles.append([p for ext in (".npz",".wav",".mp4",".json") if os.path.isfile(p:=stem+ext)])\ndef consume(paths):\n    total=0\n    for path in paths:\n        with open(path,"rb",buffering=0) as f:\n            while chunk:=f.read(8*1024*1024): total+=len(chunk)\n    return total\nt0=time.perf_counter()\nwith cf.ProcessPoolExecutor(max_workers=4) as ex: totals=list(ex.map(consume,bundles))\ndt=time.perf_counter()-t0; total=sum(totals)\nprint("participants",len(bundles),"files",sum(map(len,bundles)),"bytes",total,"seconds",f"{dt:.6f}","aggregate_MiB_per_s",f"{total/dt/2**20:.3f}")'
```

### Bounded reconnaissance, overlays, harness, and tests

```bash
VIBES_PY=/simurgh/group/lw29/conda/envs/ViBES/bin/python

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$VIBES_PY" scripts/make_dev_sample.py --config configs/recon.yaml

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$VIBES_PY" scripts/probe_media.py --config configs/recon.yaml \
  --output outputs/recon/media_probe.parquet --workers 4

PYTHONDONTWRITEBYTECODE=1 "$VIBES_PY" scripts/audit_assumptions.py

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$VIBES_PY" scripts/analyze_validity.py \
  --manifest configs/dev_sample.csv --manifest configs/dyad_audit.csv \
  --manifest configs/movement_audit.csv --source-root seamless_interaction \
  --out-dir outputs/recon/validity_audit

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$VIBES_PY" scripts/render_invalid_overlays.py \
  --candidates configs/invalid_clip_candidates.csv --max-height 960

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$VIBES_PY" scripts/render_invalid_overlays.py \
  --candidates configs/test_clean_span_candidate.csv --max-height 960

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  "$VIBES_PY" -m seamless_curation --config configs/harness.yaml --limit 120

PYTHONPATH=src "$VIBES_PY" scripts/summarize_harness.py \
  --root outputs/harness --expected-files 120

PYTHONPATH=src "$VIBES_PY" -m pytest -q
bash -n slurm/extract_dev_array.sbatch
```

The Slurm array was inspected but not submitted. To submit it in a later
session after rechecking queue/quota state: `sbatch slurm/extract_dev_array.sbatch`.
All Git commands in this environment use
`git --git-dir=.git-session --work-tree=.`.

## 9. Session-2 decisions and constraints

### Design-document assumptions refuted by observation

1. The staged layout is not flat modality directories and has no separate
   transcript/VAD JSONL or official metadata CSVs.
2. `movement_v4:is_occluded`/`pred_vertices` are absent from all 184 bounded
   files, and basic movement exists only for V00.
3. Released video is neither universally 1080p nor exactly 30 fps; nominal
   `N/30` duration mismatch creates vendor-correlated drift.
4. Four sibling paths do not imply usable content: two complete-looking V01
   bundles are empty placeholders.
5. DEFLATE NPZ members are not true memory maps.
6. The keypoint third channel is not bounded to `[0,1]`, and the native
   coordinate raster varies per file.
7. Paired annotation counts aligned in this sample, but paired video counts and
   start timestamps are not universally equal.
8. The three validity masks are not interchangeable: movement is block-zero
   feature availability, box is zero-filled visibility/tracking failure, and
   invalid SMPL-H is finite/varying rather than a universal gap encoding.

Canonical beta, metric translation units, exact coordinate-frame semantics,
and wrist-relative hand post-processing are **unverified**, not listed as
refuted.

### Decisions needed before Session 2

1. Provide or locate the official metadata tables, or approve proceeding
   without activity, relationship, participant-integrity, and recording-strata
   joins.
2. Decide whether a correctly licensed SMPL-H asset and authoritative template,
   beta, unit, and joint convention can be staged. Without them, rename/retain
   only explicitly raw/provisional motion measurements rather than claiming
   metric FK features.
3. Confirm whether V00-only movement is expected release versioning or an
   incomplete local stage, and define how unavailable movement should be
   represented in later multi-signal logic. It must not default to valid.
4. Choose a writable high-capacity target for curated media; the repository
   export reports only about 11 GiB available.
5. Decide the later timing policy for mixed native frame rates and small
   annotation/video count offsets. This session measured them and performed no
   resampling, interpolation, or repair.
6. Obtain upstream definitions for `smplh:is_valid`, `movement:is_valid`, and
   the keypoint third channel if possible; their observed behavior is not enough
   to assign formal semantics.

### Expensive or impossible with current cluster/staging

1. Metric SMPL-H FK, bone-length checks, mesh checks, and proof of hand-to-wrist
   transforms are impossible from the staged payload alone; required licensed
   assets/provenance are absent.
2. Official activity/prompt/relationship/participant audits are impossible
   until the missing tables are supplied.
3. A media-bearing curated subset will not fit the current approximately
   11-GiB repository export; a different writable volume is required.
4. Full-corpus passes will be NFS-bound. Cache-uncontrolled reads measured
   26 MiB/s single-worker and 114 MiB/s at four workers; a small warm benchmark
   was about 80 times faster and must not be used for capacity planning.
5. Slurm arrays are limited to 1,001 indices, so approximately 130k files need
   batched arrays or a task manifest; the exact filesystem hard/soft quota is
   still unverified because quota clients are unavailable.
6. Outbound network worked on the current interactive node, but is unverified
   for future batch nodes. Any later package/checkpoint dependency needs a
   small connectivity check and a cluster-local cache before jobs are designed
   around it.
