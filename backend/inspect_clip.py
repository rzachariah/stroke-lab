"""
Inspection harness for the OTI pipeline — local testing, no Claude call.

Runs extract_poses -> compute_oti_metrics on one or more videos and prints the
RAW signal so we can judge whether the metrics are trustworthy:
  - pose detection coverage (how many frames got a confident skeleton)
  - per-frame X-factor / knee / wrist series (with simple noise stats)
  - detected contact frame & peak-rotation frame
  - the final report dict (what coach.py would receive)

Usage:
  source venv/bin/activate
  python inspect_clip.py samples/clip1.mov --side r --stroke forehand
  python inspect_clip.py samples/*.mov            # batch, summary table
  python inspect_clip.py samples/clip1.mov --full # dump every frame
"""
import argparse
import glob
import json
import statistics
import sys

from analysis.pose import extract_poses
from analysis.oti import compute_oti_metrics


def _series(frames, attr):
    return [getattr(f, attr) for f in frames if getattr(f, attr) is not None]


def _noise(vals):
    """Crude jitter measure: mean absolute frame-to-frame delta."""
    if len(vals) < 2:
        return 0.0
    return statistics.fmean(abs(vals[i] - vals[i - 1]) for i in range(1, len(vals)))


def inspect(path: str, side: str, stroke: str, sample_every_n: int, full: bool) -> dict:
    print(f"\n{'=' * 70}\n{path}  (side={side}, stroke={stroke}, sample_every_n={sample_every_n})\n{'=' * 70}")
    seq = extract_poses(path, sample_every_n=sample_every_n)
    n = len(seq.frames)
    if n == 0:
        print("  !! no frames read — bad path or unreadable codec")
        return {"path": path, "error": "no frames"}

    detected = sum(1 for f in seq.frames if f.landmarks is not None)
    world_detected = sum(1 for f in seq.frames if f.world_landmarks is not None)
    dur = seq.frames[-1].time_sec
    print(f"  fps={seq.fps:.1f}  sampled_frames={n}  duration={dur:.2f}s")
    print(f"  pose coverage: 2D={detected}/{n} ({100*detected/n:.0f}%)  "
          f"world={world_detected}/{n} ({100*world_detected/n:.0f}%)")

    report = compute_oti_metrics(seq, player_side=side)

    xf = _series(report.frames, "x_factor")
    lk = _series(report.frames, "l_knee_bend")
    rk = _series(report.frames, "r_knee_bend")
    wr = _series(report.frames, "wrist_x_rel")

    def stat_line(name, vals, unit=""):
        if not vals:
            print(f"  {name:14s}: (none detected)")
            return
        print(f"  {name:14s}: n={len(vals):3d}  min={min(vals):7.2f}  "
              f"max={max(vals):7.2f}  mean={statistics.fmean(vals):7.2f}  "
              f"jitter/frame={_noise(vals):6.2f}{unit}")

    print("  --- raw metric series ---")
    stat_line("x_factor", xf, "°")
    stat_line("l_knee_bend", lk, "°")
    stat_line("r_knee_bend", rk, "°")
    stat_line("wrist_x_rel", wr)

    print("  --- detected events ---")
    print(f"  contact_frame_idx   = {report.contact_frame_idx} "
          f"(of {len(report.frames)} sampled)")
    print(f"  peak_x_factor       = {report.peak_x_factor}° at t={report.peak_x_factor_time}s")
    print(f"  swing_direction     = {report.swing_direction!r}  "
          f"{'<-- STUB, always neutral' if report.swing_direction == 'neutral' else ''}")

    print("  --- report dict (what coach.py receives) ---")
    print("  " + json.dumps(report.to_dict(), indent=2).replace("\n", "\n  "))

    if full:
        print("  --- per-frame ---")
        print(f"  {'t':>6} {'xfac':>7} {'lknee':>7} {'rknee':>7} {'wristx':>8}")
        for f in report.frames:
            print(f"  {f.time_sec:6.2f} "
                  f"{(f.x_factor if f.x_factor is not None else float('nan')):7.1f} "
                  f"{(f.l_knee_bend if f.l_knee_bend is not None else float('nan')):7.1f} "
                  f"{(f.r_knee_bend if f.r_knee_bend is not None else float('nan')):7.1f} "
                  f"{(f.wrist_x_rel if f.wrist_x_rel is not None else float('nan')):8.3f}")

    return {
        "path": path,
        "coverage_pct": round(100 * detected / n),
        "peak_x_factor": report.peak_x_factor,
        "min_knee_bend": report.min_knee_bend,
        "contact_wrist_x_rel": report.contact_wrist_x_rel,
        "swing_direction": report.swing_direction,
        "scores": {"legs": report.leg_score, "shoulders": report.shoulder_score,
                   "late_hit": report.late_hit_score},
        "x_factor_jitter": round(_noise(xf), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="video paths or globs")
    ap.add_argument("--side", default="r", choices=["r", "l"])
    ap.add_argument("--stroke", default="forehand")
    ap.add_argument("--sample-every-n", type=int, default=1,
                    help="1 = full framerate (default here; routes.py uses 2)")
    ap.add_argument("--full", action="store_true", help="dump every frame")
    args = ap.parse_args()

    paths = []
    for v in args.videos:
        paths.extend(sorted(glob.glob(v)) or [v])

    summary = []
    for p in paths:
        summary.append(inspect(p, args.side, args.stroke, args.sample_every_n, args.full))

    if len(summary) > 1:
        print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
        for s in summary:
            if "error" in s:
                print(f"  {s['path']}: ERROR {s['error']}")
                continue
            sc = s["scores"]
            print(f"  {s['path'].split('/')[-1]:24s} cov={s['coverage_pct']:3d}%  "
                  f"xfac={s['peak_x_factor']}°(jit {s['x_factor_jitter']})  "
                  f"knee={s['min_knee_bend']}°  "
                  f"scores L/S/LH={sc['legs']}/{sc['shoulders']}/{sc['late_hit']}  "
                  f"dir={s['swing_direction']}")


if __name__ == "__main__":
    sys.exit(main())
