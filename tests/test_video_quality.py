"""Tests for the per-clip pixel pass.

The synthetic video is built so every measure has a known answer: a textured
background that jumps 12 px sideways at 6 s (a camera bump), a subject patch that
is sharp noise except for 3-6 s where it is blurred, and a 2.5 s GOP so that
some clips' keyframe-before-midpoint falls before the clip start and has to be
decoded forward from. The same video stored quarter-turned must measure the same
once the raster repair stands it upright.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")

from seamless_curation.media_repair import RasterRepair, repair_box
from seamless_curation.video_quality import (
    VISUAL_COLUMNS,
    ClipWindow,
    UprightFrame,
    background_shift,
    empty_measures,
    frame_measures,
    map_box,
    measure_video,
    upright_gray,
)

FPS = 10
GOP = 25  # keyframes at 0, 2.5, 5, 7.5, 10 s
WIDTH, HEIGHT = 540, 960
N_FRAMES = 105  # 10.5 s
SHIFT_AT_S = 6.0
SHIFT_PX = 12
BLUR_SPAN_S = (3.0, 6.0)
BOX_EARLY = (170, 300, 370, 700)
BOX_LATE = (180, 300, 380, 700)

WINDOWS = [
    ClipWindow(0, 0.0, 3.0),    # keyframe 0 s, inside
    ClipWindow(1, 3.0, 6.0),    # keyframe 2.5 s < start: decode forward to 3.0
    ClipWindow(2, 6.0, 9.0),    # keyframe 7.5 s = midpoint
    ClipWindow(3, 9.0, 10.5),   # tail: keyframe 7.5 s < start: forward to 9.0
    ClipWindow(4, 12.0, 15.0),  # past the end of the video
]

_rng = np.random.default_rng(20260924)
_BACKGROUND = cv2.GaussianBlur(
    _rng.uniform(0, 255, (HEIGHT + 40, WIDTH + 40)).astype(np.float32), (0, 0), 3
)
_BACKGROUND = cv2.normalize(_BACKGROUND, None, 40, 215, cv2.NORM_MINMAX).astype(np.uint8)
_SHARP = _rng.uniform(40, 215, (400, 200)).astype(np.uint8)
_BLURRED = cv2.GaussianBlur(_SHARP, (0, 0), 4)


def _box_for(t: float) -> tuple[int, int, int, int]:
    return BOX_LATE if t >= SHIFT_AT_S else BOX_EARLY


def _upright_frame(k: int) -> np.ndarray:
    t = k / FPS
    dx = SHIFT_PX if t >= SHIFT_AT_S else 0
    image = _BACKGROUND[20:20 + HEIGHT, 20 + dx:20 + dx + WIDTH].copy()
    x1, y1, x2, y2 = _box_for(t)
    image[y1:y2, x1:x2] = _BLURRED if BLUR_SPAN_S[0] <= t < BLUR_SPAN_S[1] else _SHARP
    return image


def _write_video(path: Path, frames: list[np.ndarray]) -> Path:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.height, stream.width = frames[0].shape
    stream.pix_fmt = "yuv420p"
    stream.options = {
        "crf": "4",
        "x264-params": f"keyint={GOP}:min-keyint={GOP}:scenecut=0:bframes=2",
    }
    for image in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="gray")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture(scope="module")
def videos(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("video_quality")
    upright = [_upright_frame(k) for k in range(N_FRAMES)]
    turned = [cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) for image in upright]
    return {
        "upright": _write_video(root / "upright.mp4", upright),
        "turned": _write_video(root / "turned.mp4", turned),
    }


UPRIGHT = RasterRepair(WIDTH, HEIGHT)
# Stored clockwise-turned, so one counter-clockwise quarter turn stands it up.
TURNED = RasterRepair(HEIGHT, WIDTH, quarter_turns=1)


def _upright_box_at(t: float) -> np.ndarray:
    return np.asarray(_box_for(t), dtype=np.float64)


def _turned_box_at(t: float) -> np.ndarray:
    # upright (x, y) -> stored (HEIGHT - 1 - y, x) under a clockwise turn
    x1, y1, x2, y2 = _box_for(t)
    return np.array([HEIGHT - 1 - y2, x1, HEIGHT - 1 - y1, x2], dtype=np.float64)


@pytest.fixture(scope="module")
def measured(videos) -> list[dict]:
    return measure_video(videos["upright"], WINDOWS, repair=UPRIGHT, box_at=_upright_box_at)


def test_statuses_times_and_columns(measured):
    assert [row["clip_index"] for row in measured] == [0, 1, 2, 3, 4]
    assert [row["visual_status"] for row in measured] == ["measured"] * 4 + ["no_frame"]
    for row in measured:
        assert set(row) == {"clip_index", *VISUAL_COLUMNS}
    times = [row["frame_time_s"] for row in measured[:4]]
    # keyframe before the midpoint when it is inside the clip, else the clip's first frame
    assert times == pytest.approx([0.0, 3.0, 7.5, 9.0], abs=1e-3)
    for window, row in zip(WINDOWS[:4], measured[:4]):
        assert window.start_s - 1e-3 <= row["frame_time_s"] < window.end_s
    unmeasured = measured[4]
    assert all(math.isnan(unmeasured[name]) for name in VISUAL_COLUMNS if name != "visual_status")


def test_blurred_subject_scores_lower_sharpness(measured):
    sharp, blurred, sharp_again = (measured[i]["frame_sharpness"] for i in (0, 1, 2))
    assert blurred < sharp / 3
    assert blurred < sharp_again / 3


def test_camera_bump_is_a_shift_and_a_fixed_camera_is_not(measured):
    shifts = [row["camera_shift_px"] for row in measured]
    assert math.isnan(shifts[0])  # first clip: nothing to compare with
    assert shifts[1] < 1.0        # same background (the subject changed, but it is masked)
    assert shifts[2] == pytest.approx(SHIFT_PX, abs=1.5)
    assert shifts[3] < 1.0
    assert math.isnan(shifts[4])


def test_frame_level_measures_are_in_range(measured):
    for row in measured[:4]:
        assert 40 <= row["frame_luma_mean"] <= 215
        assert 0.0 <= row["frame_luma_clipped_frac"] < 0.01
        assert 0.0 <= row["background_edge_density"] <= 1.0


def test_quarter_turned_storage_measures_like_upright(videos, measured):
    turned = measure_video(videos["turned"], WINDOWS, repair=TURNED, box_at=_turned_box_at)
    assert [row["visual_status"] for row in turned] == [row["visual_status"] for row in measured]
    for a, b in zip(measured[:4], turned[:4]):
        assert b["frame_time_s"] == pytest.approx(a["frame_time_s"], abs=1e-3)
        assert b["frame_sharpness"] == pytest.approx(a["frame_sharpness"], rel=0.05)
        assert b["frame_luma_mean"] == pytest.approx(a["frame_luma_mean"], abs=1.0)
        assert b["background_edge_density"] == pytest.approx(a["background_edge_density"], abs=0.01)
    assert turned[2]["camera_shift_px"] == pytest.approx(SHIFT_PX, abs=1.5)


def test_window_order_is_preserved_and_shift_follows_time(videos, measured):
    reversed_windows = WINDOWS[::-1]
    out = measure_video(videos["upright"], reversed_windows, repair=UPRIGHT, box_at=_upright_box_at)
    assert [row["clip_index"] for row in out] == [4, 3, 2, 1, 0]
    by_index = {row["clip_index"]: row for row in out}
    for row in measured:
        other = by_index[row["clip_index"]]
        for name in VISUAL_COLUMNS:
            if isinstance(row[name], float) and math.isnan(row[name]):
                assert math.isnan(other[name])
            else:
                assert other[name] == pytest.approx(row[name])


def test_no_box_leaves_box_measures_na_but_frame_measured(videos):
    out = measure_video(videos["upright"], WINDOWS[:2], repair=UPRIGHT, box_at=lambda t: None)
    for row in out:
        assert row["visual_status"] == "measured"
        assert math.isnan(row["frame_sharpness"])
        assert math.isnan(row["background_edge_density"])
        assert np.isfinite(row["frame_luma_mean"])
    assert np.isfinite(out[1]["camera_shift_px"])  # nothing to mask, background still static
    assert out[1]["camera_shift_px"] < 1.0


def test_error_marks_the_window_and_the_rest_decode_error(videos):
    calls = {"n": 0}

    def flaky_box_at(t: float) -> np.ndarray:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return _upright_box_at(t)

    out = measure_video(videos["upright"], WINDOWS, repair=UPRIGHT, box_at=flaky_box_at)
    assert [row["visual_status"] for row in out] == ["measured", "measured"] + ["decode_error"] * 3


def test_wrong_raster_is_a_decode_error_not_a_crash(videos):
    out = measure_video(
        videos["upright"], WINDOWS[:2], repair=RasterRepair(1080, 1920), box_at=_upright_box_at
    )
    assert [row["visual_status"] for row in out] == ["decode_error", "decode_error"]


def test_missing_and_corrupt_files_never_raise(tmp_path):
    missing = measure_video(tmp_path / "absent.mp4", WINDOWS, repair=UPRIGHT, box_at=_upright_box_at)
    assert [row["visual_status"] for row in missing] == ["decode_error"] * len(WINDOWS)
    assert [row["clip_index"] for row in missing] == [w.clip_index for w in WINDOWS]
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"\x00\x01not a video" * 100)
    corrupt = measure_video(garbage, WINDOWS, repair=UPRIGHT, box_at=_upright_box_at)
    assert [row["visual_status"] for row in corrupt] == ["decode_error"] * len(WINDOWS)
    assert measure_video(garbage, [], repair=UPRIGHT, box_at=_upright_box_at) == []


def test_empty_measures_row():
    row = empty_measures()
    assert row["visual_status"] == "not_run"
    assert set(row) == set(VISUAL_COLUMNS)
    assert all(math.isnan(row[name]) for name in VISUAL_COLUMNS if name != "visual_status")
    with pytest.raises(ValueError):
        empty_measures("measured")


def test_exposure_measures():
    dark = np.full((960, 540), 3, np.uint8)
    dark[:, :54] = 128
    out = frame_measures(dark, (100, 100, 300, 500))
    assert out["frame_luma_clipped_frac"] == pytest.approx(0.9)
    assert out["frame_luma_mean"] == pytest.approx(0.9 * 3 + 0.1 * 128)
    blown = np.full((960, 540), 252, np.uint8)
    assert frame_measures(blown, None)["frame_luma_clipped_frac"] == 1.0
    mid = np.full((960, 540), 128, np.uint8)
    assert frame_measures(mid, None)["frame_luma_clipped_frac"] == 0.0


def test_edge_density_counts_only_background():
    plain = np.full((960, 540), 128, np.uint8)
    busy_subject = plain.copy()
    busy_subject[300:700, 170:370] = _SHARP
    box = (170, 300, 370, 700)
    assert frame_measures(busy_subject, box)["background_edge_density"] == 0.0
    cluttered = _BACKGROUND[:960, :540].copy()
    cluttered[::40, :] = 250  # lines = edges
    assert frame_measures(cluttered, box)["background_edge_density"] > 0.01
    # a valid box that lies off the frame leaves the whole frame as background
    off = frame_measures(cluttered, (540, 0, 540, 960))
    assert math.isnan(off["frame_sharpness"])
    whole = cv2.Canny(cluttered, 100, 200) > 0
    assert off["background_edge_density"] == pytest.approx(whole.mean())


def test_background_shift_edge_cases():
    image = _BACKGROUND[:960, :540].copy()
    frame = UprightFrame(0.0, image, BOX_EARLY)
    assert background_shift(frame, UprightFrame(1.0, image.copy(), BOX_EARLY)) < 0.1
    flat = UprightFrame(1.0, np.full_like(image, 90), BOX_EARLY)
    assert math.isnan(background_shift(frame, flat))
    smaller = UprightFrame(1.0, image[:900], BOX_EARLY)
    assert math.isnan(background_shift(frame, smaller))
    covered = UprightFrame(1.0, image.copy(), (0, 0, 540, 960))
    assert math.isnan(background_shift(frame, covered))


@pytest.mark.parametrize(
    "repair",
    [
        RasterRepair(1080, 1920),
        RasterRepair(2160, 2160, sample_aspect=0.5625),
        RasterRepair(1080, 960, pillarbox=(270, 270)),
        RasterRepair(3840, 2160, quarter_turns=1),
        RasterRepair(640, 480, quarter_turns=3),
    ],
    ids=["square", "anamorphic", "pillarbox", "turned_ccw", "turned_cw_upscaled"],
)
def test_box_lands_on_the_same_pixels_after_repair_and_scaling(repair):
    stored = np.zeros((repair.height, repair.width), np.uint8)
    left = repair.pillarbox[0]
    x1, x2 = left + int(0.30 * (repair.width - sum(repair.pillarbox))), left + int(0.55 * (repair.width - sum(repair.pillarbox)))
    y1, y2 = int(0.20 * repair.height), int(0.70 * repair.height)
    stored[y1:y2, x1:x2] = 255
    image, sx, sy = upright_gray(stored, repair, 540)
    assert min(image.shape) == 540
    assert image.shape[::-1] == tuple(
        int(round(d * 540 / min(repair.display_size))) for d in repair.display_size
    )
    box = map_box(np.array([x1, y1, x2, y2], float), repair, (sx, sy), image.shape)
    ys, xs = np.nonzero(image > 127)
    lit = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    assert np.allclose(box, lit, atol=2.5), (box, lit)
    # mapping through the module agrees with media_repair's own box transform
    raw = repair_box(np.array([x1, y1, x2, y2], float), repair)
    assert box[0] == pytest.approx(raw[0] * sx, abs=1.0)
    assert box[3] == pytest.approx(raw[3] * sy, abs=1.0)


def test_map_box_handles_missing_and_offscreen():
    assert map_box(None, UPRIGHT, (1.0, 1.0), (960, 540)) is None
    assert map_box(np.array([np.nan, 0, 10, 10]), UPRIGHT, (1.0, 1.0), (960, 540)) is None
    x1, y1, x2, y2 = map_box(np.array([-50.0, -20, 600, 2000]), UPRIGHT, (1.0, 1.0), (960, 540))
    assert (x1, y1, x2, y2) == (0, 0, 540, 960)
    off = map_box(np.array([700.0, 10, 800, 20]), UPRIGHT, (1.0, 1.0), (960, 540))
    assert off[2] <= off[0]
