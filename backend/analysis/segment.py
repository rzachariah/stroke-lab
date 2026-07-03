"""
Stroke segmentation — split a long session clip into individual strokes.

A long video (e.g. 6 min of baseline practice) contains many strokes plus
walking, ball pickups, and standing around. Computing OTI metrics over the
whole clip is meaningless. This module finds the individual swings so each can
be analyzed separately.

Approach: during a swing the dominant-hand wrist moves fast, then decelerates.
We compute dominant-wrist speed across the clip, find prominent peaks (one per
swing) with a refractory gap so a single swing isn't double-counted, and return
a time window around each peak.
"""
from dataclasses import dataclass
import numpy as np

from .pose import PoseSequence


@dataclass
class StrokeWindow:
    index: int          # 0-based stroke number within the clip
    peak_time: float    # time of max wrist speed (≈ contact / swing apex)
    start_time: float
    end_time: float
    start_i: int        # indices into seq.frames
    end_i: int
    peak_speed: float
    uprightness: float = 0.0   # median torso verticality over the window, 0..1
    is_stroke: bool = True     # passes the non-stroke filter


def _dominant_wrist_speed(seq: PoseSequence, player_side: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame speed of the dominant wrist in normalized image units / second.

    Uses normalized (resolution-independent) landmark coords so it works
    regardless of how big the player is in frame. Frames with no confident
    wrist get speed 0 (treated as not-swinging)."""
    name = f"{player_side}_wrist"
    times, pos = [], []
    for f in seq.frames:
        lm = f.landmarks.get(name) if f.landmarks else None
        if lm is not None and lm[3] >= 0.5:
            pos.append((lm[0], lm[1]))
        else:
            pos.append(None)
        times.append(f.time_sec)

    times = np.asarray(times)
    speed = np.zeros(len(seq.frames))
    for i in range(1, len(pos)):
        a, b = pos[i - 1], pos[i]
        dt = times[i] - times[i - 1]
        if a is None or b is None or dt <= 0:
            continue
        speed[i] = np.hypot(b[0] - a[0], b[1] - a[1]) / dt

    # light smoothing (3-tap) to suppress single-frame jitter
    if len(speed) >= 3:
        speed = np.convolve(speed, np.ones(3) / 3, mode="same")
    return times, speed


def _torso_uprightness(frame) -> float | None:
    """How vertical the shoulder->hip torso axis is, in [0,1] (1 = upright).

    A swinging player stands upright; someone bending into a ball basket tilts
    the torso toward horizontal. Uses normalized image coords (resolution-free).
    Returns None if shoulders/hips aren't both visible."""
    if not frame.landmarks:
        return None

    def mid(a, b):
        la, lb = frame.landmarks.get(a), frame.landmarks.get(b)
        if la is None or lb is None or la[3] < 0.5 or lb[3] < 0.5:
            return None
        return np.array([(la[0] + lb[0]) / 2, (la[1] + lb[1]) / 2])

    sh, hip = mid("l_shoulder", "r_shoulder"), mid("l_hip", "r_hip")
    if sh is None or hip is None:
        return None
    vec = sh - hip
    n = np.hypot(vec[0], vec[1])
    if n < 1e-6:
        return None
    return float(abs(vec[1]) / n)   # |dy|/length: 1 = perfectly vertical torso


def find_stroke_windows(
    seq: PoseSequence,
    player_side: str = "r",
    pre_sec: float = 0.5,
    post_sec: float = 0.5,
    min_gap_sec: float = 0.8,
    prominence_k: float = 2.0,
    min_uprightness: float = 0.85,
) -> list[StrokeWindow]:
    """Detect stroke windows from dominant-wrist speed peaks.

    A candidate is a frame whose speed exceeds median + prominence_k*MAD. Among
    candidates we greedily keep the fastest, suppressing any others within
    min_gap_sec (non-max suppression) so one swing yields one window.
    """
    if len(seq.frames) < 3:
        return []

    times, speed = _dominant_wrist_speed(seq, player_side)

    # robust threshold from the moving speed (ignore the many ~0 standing frames)
    moving = speed[speed > np.percentile(speed, 50)]
    if moving.size == 0:
        return []
    med = np.median(moving)
    mad = np.median(np.abs(moving - med)) or (moving.std() or 1e-6)
    thr = med + prominence_k * mad

    cand = [i for i in range(1, len(speed) - 1)
            if speed[i] >= thr and speed[i] >= speed[i - 1] and speed[i] >= speed[i + 1]]
    cand.sort(key=lambda i: speed[i], reverse=True)

    kept: list[int] = []
    for i in cand:
        if all(abs(times[i] - times[j]) >= min_gap_sec for j in kept):
            kept.append(i)
    kept.sort()

    windows = []
    for n, pk in enumerate(kept):
        t0, t1 = times[pk] - pre_sec, times[pk] + post_sec
        start_i = max(0, int(np.searchsorted(times, t0)))
        end_i = min(len(seq.frames), int(np.searchsorted(times, t1)) + 1)
        up = [u for u in (_torso_uprightness(f) for f in seq.frames[start_i:end_i])
              if u is not None]
        upright = float(np.median(up)) if up else 0.0
        windows.append(StrokeWindow(
            index=n, peak_time=round(float(times[pk]), 2),
            start_time=round(float(times[start_i]), 2),
            end_time=round(float(times[min(end_i, len(seq.frames)) - 1]), 2),
            start_i=start_i, end_i=end_i, peak_speed=round(float(speed[pk]), 3),
            uprightness=round(upright, 3),
            is_stroke=upright >= min_uprightness,
        ))
    # renumber kept strokes 0..K-1 so the stroke index is meaningful to a user
    return windows


def subsequence(seq: PoseSequence, w: StrokeWindow) -> PoseSequence:
    """Build a PoseSequence containing only the frames of one stroke window."""
    sub = PoseSequence(fps=seq.fps)
    sub.frames = seq.frames[w.start_i:w.end_i]
    return sub
