"""
Run stroke segmentation + per-stroke OTI metrics on a session video.
No Claude call. Prints a per-stroke table and aggregate stats.

  source venv/bin/activate
  python segment_run.py /path/to/session.MP4 --side r --every 2 --complexity 1
"""
import argparse
import json
import os
import statistics

from analysis.pose import extract_poses
from analysis.oti import compute_oti_metrics
from analysis.segment import find_stroke_windows, subsequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--side", default="r", choices=["r", "l"])
    ap.add_argument("--stroke", default="forehand")
    ap.add_argument("--every", type=int, default=2, help="sample_every_n for pose extraction")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    args = ap.parse_args()

    print(f"extracting poses (every={args.every}, complexity={args.complexity}) ...", flush=True)
    seq = extract_poses(args.video, sample_every_n=args.every, model_complexity=args.complexity)
    print(f"  {len(seq.frames)} pose frames, fps={seq.fps:.0f}, "
          f"dur={seq.frames[-1].time_sec:.0f}s" if seq.frames else "  no frames", flush=True)

    windows = find_stroke_windows(seq, player_side=args.side)
    kept = [w for w in windows if w.is_stroke]
    rejected = [w for w in windows if not w.is_stroke]
    print(f"\n{len(windows)} candidate windows -> {len(kept)} strokes, "
          f"{len(rejected)} rejected as non-strokes (low torso uprightness)\n")

    print(f"  {'#':>3} {'t_peak':>7} {'up':>5} {'xfac':>6} {'knee':>6} {'late':>6} {'L/S/LH':>12}")
    rows = []
    for w in kept:
        rep = compute_oti_metrics(subsequence(seq, w), player_side=args.side)
        rows.append((w, rep))
        sc = f"{rep.leg_score}/{rep.shoulder_score}/{rep.late_hit_score}"
        print(f"  {w.index:>3} {w.peak_time:>7.1f} {w.uprightness:>5.2f} "
              f"{str(rep.peak_x_factor):>6} {str(rep.min_knee_bend):>6} "
              f"{str(rep.contact_wrist_x_rel):>6} {sc:>12}")

    if rejected:
        print(f"\n  rejected windows (t_peak / uprightness):")
        print("   " + "  ".join(f"{w.peak_time:.0f}s({w.uprightness:.2f})" for w in rejected))

    def agg(vals):
        vals = [v for v in vals if v is not None]
        return (round(statistics.fmean(vals), 1), round(statistics.median(vals), 1)) if vals else (None, None)

    print(f"\n  --- aggregate over {len(rows)} real strokes (mean, median) ---")
    print(f"  peak_x_factor : {agg([r.peak_x_factor for _, r in rows])}")
    print(f"  min_knee_bend : {agg([r.min_knee_bend for _, r in rows])}")
    print(f"  leg_score     : {agg([r.leg_score for _, r in rows])}")
    print(f"  shoulder_score: {agg([r.shoulder_score for _, r in rows])}")
    print(f"  late_hit_score: {agg([r.late_hit_score for _, r in rows])}")

    out = f"inspect_out/{os.path.basename(args.video)}.strokes.json"
    with open(out, "w") as f:
        json.dump([{"window": w.__dict__, "report": r.to_dict()} for w, r in rows], f, indent=2)
    print(f"\n  wrote {out} ({len(rows)} strokes)")


if __name__ == "__main__":
    main()
