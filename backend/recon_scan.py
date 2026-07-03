"""Fast structural recon of a (possibly long) clip.

Downsamples heavily and uses model_complexity=1 to quickly answer:
  - Does MediaPipe reliably detect the player?
  - How big is the player in frame (proxy for detection reliability / distance)?
  - What does the X-factor / knee / wrist time-series look like across the clip?
  - Is this one stroke or a long multi-stroke session?

Writes a JSON of the sampled series to inspect_out/<name>.recon.json and prints
a compact ASCII timeline.
"""
import json
import math
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

from analysis.pose import L
from analysis.oti import _x_factor_3d, _knee_bend_angle
from analysis.pose import Frame

mp_pose = mp.solutions.pose


def scan(path: str, every_n: int = 10):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    name = os.path.basename(path)
    print(f"\n=== {name}  {w}x{h} fps={fps:.0f} frames={total} dur={total/fps:.0f}s "
          f"(sampling every {every_n} -> ~{total//every_n} poses) ===", flush=True)

    rows = []
    idx = 0
    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, bgr = cap.read()
            if not ret:
                break
            if idx % every_n == 0:
                res = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                lm = None
                wlm = None
                psize = None
                if res.pose_landmarks:
                    lm = {n: (res.pose_landmarks.landmark[i].x,
                              res.pose_landmarks.landmark[i].y,
                              res.pose_landmarks.landmark[i].z,
                              res.pose_landmarks.landmark[i].visibility)
                          for n, i in L.items()}
                    # person size = normalized vertical span shoulder->ankle (0..1 of frame h)
                    sh = res.pose_landmarks.landmark[L["r_shoulder"]].y
                    ak = res.pose_landmarks.landmark[L["r_ankle"]].y
                    psize = abs(ak - sh)
                if res.pose_world_landmarks:
                    wlm = {n: (res.pose_world_landmarks.landmark[i].x,
                               res.pose_world_landmarks.landmark[i].y,
                               res.pose_world_landmarks.landmark[i].z,
                               res.pose_world_landmarks.landmark[i].visibility)
                           for n, i in L.items()}
                fr = Frame(index=idx, time_sec=idx / fps, landmarks=lm,
                           world_landmarks=wlm, width=w, height=h)
                rows.append({
                    "t": round(idx / fps, 2),
                    "det": lm is not None,
                    "psize": round(psize, 3) if psize else None,
                    "xf": round(_x_factor_3d(fr), 1) if _x_factor_3d(fr) is not None else None,
                    "lk": round(_knee_bend_angle(fr, "l"), 0) if _knee_bend_angle(fr, "l") is not None else None,
                    "rk": round(_knee_bend_angle(fr, "r"), 0) if _knee_bend_angle(fr, "r") is not None else None,
                })
            idx += 1
    cap.release()

    det = [r for r in rows if r["det"]]
    print(f"  detection: {len(det)}/{len(rows)} sampled poses ({100*len(det)/max(1,len(rows)):.0f}%)")
    if det:
        ps = [r["psize"] for r in det if r["psize"]]
        if ps:
            print(f"  player size in frame (shoulder->ankle, frac of height): "
                  f"min={min(ps):.2f} median={sorted(ps)[len(ps)//2]:.2f} max={max(ps):.2f}")
        xf = [r["xf"] for r in det if r["xf"] is not None]
        if xf:
            print(f"  x_factor over whole clip: min={min(xf):.0f} median={sorted(xf)[len(xf)//2]:.0f} max={max(xf):.0f}")

    # ASCII timeline of detection (each char ~ one sampled frame bucket)
    buckets = 70
    if rows:
        per = max(1, len(rows) // buckets)
        line = ""
        for b in range(0, len(rows), per):
            chunk = rows[b:b + per]
            frac = sum(1 for r in chunk if r["det"]) / len(chunk)
            line += "#" if frac > 0.8 else ("+" if frac > 0.4 else ".")
        span = len(rows) / fps * every_n
        print(f"  detection timeline (0..{span:.0f}s):  #=>80% +=>40% .=low")
        print("  [" + line + "]")

    out = f"inspect_out/{name}.recon.json"
    with open(out, "w") as f:
        json.dump(rows, f)
    print(f"  wrote {out}")
    return rows


if __name__ == "__main__":
    every = int(os.environ.get("EVERY_N", "10"))
    for p in sys.argv[1:]:
        scan(p, every_n=every)
