"""
OTI (Optimum Tennis Instruction) metrics.

Three power sources analyzed:
  1. Legs       — knee bend depth and drive timing
  2. Shoulders  — X-factor (shoulder-hip separation) and rotation speed
  3. Late hit   — contact point position relative to front hip
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .pose import PoseSequence, Frame


def _xz_angle_deg(v: np.ndarray) -> float:
    """Angle of a 3D vector projected onto the horizontal (X-Z) plane, in degrees."""
    return math.degrees(math.atan2(v[2], v[0]))


def _x_factor_3d(frame: Frame) -> Optional[float]:
    """
    X-factor: angle between the shoulder axis and hip axis in the horizontal plane,
    computed from MediaPipe world landmarks (metric 3D, hip-centered).

    In MediaPipe world coords: x = person's right, y = up, z = toward camera.
    We project each axis onto the xz plane and measure the angle between them.
    Returns a value in [0, 90] degrees. Returns None if landmarks are unavailable.
    """
    l_hip_w = frame.world("l_hip")
    r_hip_w = frame.world("r_hip")
    l_sh_w = frame.world("l_shoulder")
    r_sh_w = frame.world("r_shoulder")

    if any(v is None for v in [l_hip_w, r_hip_w, l_sh_w, r_sh_w]):
        return None

    hip_vec = r_hip_w - l_hip_w           # points toward person's right
    sh_vec = r_sh_w - l_sh_w

    hip_angle = _xz_angle_deg(hip_vec)
    sh_angle = _xz_angle_deg(sh_vec)

    diff = abs(sh_angle - hip_angle)
    if diff > 180:
        diff = 360 - diff
    return min(diff, 90.0)


def _knee_bend_angle(frame: Frame, side: str) -> Optional[float]:
    """Angle at the knee joint (hip-knee-ankle). 180° = straight leg."""
    hip = frame.pt(f"{side}_hip")
    knee = frame.pt(f"{side}_knee")
    ankle = frame.pt(f"{side}_ankle")
    if hip is None or knee is None or ankle is None:
        return None
    v1 = hip - knee
    v2 = ankle - knee
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(np.clip(cos_a, -1, 1)))


@dataclass
class FrameMetrics:
    time_sec: float
    x_factor: Optional[float] = None         # 3D shoulder-hip yaw separation, [0, 90]°
    l_knee_bend: Optional[float] = None      # 180 = straight, <160 = good coil
    r_knee_bend: Optional[float] = None
    wrist_x_rel: Optional[float] = None      # wrist x relative to front hip (normalized)


@dataclass
class OTIReport:
    frames: list[FrameMetrics] = field(default_factory=list)

    # Peak values
    peak_x_factor: Optional[float] = None
    peak_x_factor_time: Optional[float] = None
    min_knee_bend: Optional[float] = None       # smallest = deepest bend

    # Contact frame estimate
    contact_frame_idx: Optional[int] = None
    contact_wrist_x_rel: Optional[float] = None  # positive = in front of hip (late hit)

    # Swing direction (inside-out vs outside-in)
    swing_direction: Optional[str] = None        # "inside-out", "neutral", "outside-in"

    # Summary scores 0-10
    leg_score: Optional[float] = None
    shoulder_score: Optional[float] = None
    late_hit_score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "peak_x_factor": self.peak_x_factor,
            "peak_x_factor_time": self.peak_x_factor_time,
            "min_knee_bend": self.min_knee_bend,
            "contact_wrist_x_rel": self.contact_wrist_x_rel,
            "swing_direction": self.swing_direction,
            "leg_score": self.leg_score,
            "shoulder_score": self.shoulder_score,
            "late_hit_score": self.late_hit_score,
        }


def _detect_contact_frame(frames: list[FrameMetrics]) -> Optional[int]:
    """
    Estimate contact as the frame where wrist X reaches its forward-most point
    (maximum x relative to hip) then starts pulling back.
    """
    xs = [f.wrist_x_rel for f in frames if f.wrist_x_rel is not None]
    if not xs:
        return None
    # Find peak wrist extension
    vals = [(i, f.wrist_x_rel) for i, f in enumerate(frames) if f.wrist_x_rel is not None]
    if not vals:
        return None
    peak_i, _ = max(vals, key=lambda t: t[1])
    return peak_i


def _wrist_trajectory_direction(frames: list[FrameMetrics]) -> str:
    """
    Classify swing as inside-out, neutral, or outside-in from the wrist path.
    Uses wrist_x_rel in the approach phase (before contact).
    """
    # We need raw wrist positions for this — use a simple proxy from x_factor change
    # TODO: extend with actual 2D wrist path when using rear-view footage
    return "neutral"


def _sustained_peak(frames: list[FrameMetrics], w: int = 5,
                    min_valid: int = 3) -> tuple[Optional[float], Optional[float]]:
    """Largest X-factor the player *held*: the max over a centered rolling median
    of the per-frame X-factor series. Robust to single-frame spikes that a raw
    peak or high percentile would over-read. Returns (value, time) or (None, None).
    """
    xs = [f.x_factor for f in frames]
    ts = [f.time_sec for f in frames]
    half = w // 2
    best_v = best_t = None
    for i in range(len(xs)):
        window = [xs[j] for j in range(max(0, i - half), min(len(xs), i + half + 1))
                  if xs[j] is not None]
        if len(window) < min_valid:
            continue
        m = float(np.median(window))
        if best_v is None or m > best_v:
            best_v, best_t = m, ts[i]
    return best_v, best_t


def compute_oti_metrics(seq: PoseSequence, player_side: str = "r") -> OTIReport:
    """
    player_side: "r" for right-handed (dominant = right), "l" for lefty.
    For rear-view footage, "front hip" = the hip closest to the net =
    opposite side from dominant hand for forehand.
    """
    opp = "l" if player_side == "r" else "r"
    report = OTIReport()

    for frame in seq.frames:
        fm = FrameMetrics(time_sec=frame.time_sec)

        fm.x_factor = _x_factor_3d(frame)

        fm.l_knee_bend = _knee_bend_angle(frame, "l")
        fm.r_knee_bend = _knee_bend_angle(frame, "r")

        # Wrist position relative to front (non-dominant) hip for forehand rear view
        wrist = frame.pt(f"{player_side}_wrist")
        front_hip = frame.pt(f"{opp}_hip")
        if wrist is not None and front_hip is not None and frame.width > 0:
            fm.wrist_x_rel = (wrist[0] - front_hip[0]) / frame.width

        report.frames.append(fm)

    # Peak X-factor — use the sustained peak (max of a rolling median), i.e. the
    # largest separation the player actually HELD, not a single-frame spike. A raw
    # peak / p95 over-reads jittery swings (a couple of noisy frames look like a
    # great coil); a rolling median only rewards separation held across frames.
    pk_v, pk_t = _sustained_peak(report.frames)
    if pk_v is not None:
        report.peak_x_factor = round(pk_v, 1)
        report.peak_x_factor_time = round(pk_t, 2)

    # Min knee bend (deepest coil)
    knee_vals = []
    for f in report.frames:
        if f.l_knee_bend is not None:
            knee_vals.append(f.l_knee_bend)
        if f.r_knee_bend is not None:
            knee_vals.append(f.r_knee_bend)
    if knee_vals:
        report.min_knee_bend = round(min(knee_vals), 1)

    # Contact detection
    ci = _detect_contact_frame(report.frames)
    if ci is not None:
        report.contact_frame_idx = ci
        report.contact_wrist_x_rel = round(report.frames[ci].wrist_x_rel or 0, 3)

    report.swing_direction = _wrist_trajectory_direction(report.frames)

    # --- Scores ---
    # Leg score: 10 = knee bend < 140°, 5 = ~155°, 0 = > 170°
    if report.min_knee_bend is not None:
        report.leg_score = round(max(0, min(10, (170 - report.min_knee_bend) / 3)), 1)

    # Shoulder/X-factor score: 10 = X-factor >= 45°, 5 = ~27°, 0 = <= 10°
    if report.peak_x_factor is not None:
        report.shoulder_score = round(max(0, min(10, (report.peak_x_factor - 10) / 3.5)), 1)

    # Late hit score: positive wrist_x_rel at contact = in front = good
    if report.contact_wrist_x_rel is not None:
        report.late_hit_score = round(max(0, min(10, 5 + report.contact_wrist_x_rel * 20)), 1)

    return report
