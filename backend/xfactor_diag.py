"""
DIAGNOSTIC — why is per-stroke peak X-factor jittery rep-to-rep?

For a few chosen strokes (some scored high on shoulders, some ~0), extract the
per-FRAME X-factor across the swing window and show the raw signal + landmark
visibility + world-z spread. Tells us whether the rep-to-rep variation is real
(clean but different swings) or measurement noise (garbage landmarks), which
decide opposite fixes (contextualize vs denoise).

Run: source venv/bin/activate && python xfactor_diag.py
"""
import cv2, numpy as np, mediapipe as mp
from analysis.pose import L, _ROTATE_CODES, Frame
from analysis.oti import _x_factor_3d

VIDEO = "/Users/ranjith.zachariah/Downloads/My_recorded_video_382.MP4"
# (peak_time, reported peak_x_factor, shoulder_score) — 2 "good", 2 "zero"
TARGETS = [(68.3, 41.0, 8.9), (311.6, 39.0, 8.3), (136.3, 9.3, 0.0), (176.9, 8.9, 0.0)]
HALF = 0.6   # seconds either side of the peak


def build_frame(res, w, h, idx, t):
    lm = wlm = None
    if res.pose_landmarks:
        lm = {n: (res.pose_landmarks.landmark[i].x, res.pose_landmarks.landmark[i].y,
                  res.pose_landmarks.landmark[i].z, res.pose_landmarks.landmark[i].visibility)
              for n, i in L.items()}
    if res.pose_world_landmarks:
        wlm = {n: (res.pose_world_landmarks.landmark[i].x, res.pose_world_landmarks.landmark[i].y,
                   res.pose_world_landmarks.landmark[i].z, res.pose_world_landmarks.landmark[i].visibility)
               for n, i in L.items()}
    return Frame(index=idx, time_sec=t, landmarks=lm, world_landmarks=wlm, width=w, height=h)


def spark(vals):
    ticks = " .:-=+*#%@"
    lo, hi = 0, 90
    out = ""
    for v in vals:
        out += "·" if v is None else ticks[min(9, int((v - lo) / (hi - lo) * 9))]
    return out


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rot = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) % 360
    rcode = _ROTATE_CODES.get(rot)
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                  smooth_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

    for t, rep_xf, sh in TARGETS:
        f0 = int(round((t - HALF) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        xfs, viss, zs = [], [], []
        for k in range(int(2 * HALF * fps) + 1):
            ret, bgr = cap.read()
            if not ret:
                break
            if rcode is not None:
                bgr = cv2.rotate(bgr, rcode)
            h, w = bgr.shape[:2]
            fr = build_frame(pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                             w, h, f0 + k, (f0 + k) / fps)
            xf = _x_factor_3d(fr)
            xfs.append(round(xf, 0) if xf is not None else None)
            if fr.world_landmarks:
                vs = [fr.world_landmarks[n][3] for n in ("l_shoulder", "r_shoulder", "l_hip", "r_hip")]
                viss.append(min(vs))
                zs.append([fr.world_landmarks[n][2] for n in ("l_shoulder", "r_shoulder", "l_hip", "r_hip")])
            else:
                viss.append(None)
        got = [v for v in xfs if v is not None]
        vv = [v for v in viss if v is not None]
        print(f"\n=== t={t}s  reported peak_xf={rep_xf}  shoulder_score={sh} ===")
        print(f"  per-frame X-factor: [{spark(xfs)}]")
        print(f"  values: {got}")
        if got:
            g = sorted(got)
            print(f"  min={g[0]:.0f} median={g[len(g)//2]:.0f} p95={g[int(len(g)*0.95)]:.0f} "
                  f"max={g[-1]:.0f}  spread(max-min)={g[-1]-g[0]:.0f}")
        if vv:
            print(f"  shoulder/hip visibility: min={min(vv):.2f} median={sorted(vv)[len(vv)//2]:.2f}")
        if zs:
            zar = np.array(zs)
            print(f"  world-z std per landmark (L/R sh, L/R hip): "
                  f"{[round(float(s), 3) for s in zar.std(axis=0)]}  (depth stability; big=noisy)")
    cap.release()


if __name__ == "__main__":
    main()
