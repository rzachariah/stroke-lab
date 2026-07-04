"""
PROTOTYPE — shadow-swing vs ball-contact classifier via ball tracking.

For each detected swing window, look at frames around the swing peak, find the
dominant-hand wrist (MediaPipe) and any MOVING optic-yellow ball blobs, and measure
how close the nearest ball gets to the racket head. A struck ball sits at the racket
head (~1-2 torso-lengths past the wrist) at the peak; shadow swings have no ball there.

Two ideas make it work (see moving_balls() and the D_LO/D_HI band):
  - motion filter: only yellow present in frame f but gone at f-1 and f+1. Rejects
    static foliage/hopper/ground/held balls, which color-only detection can't.
  - contact band: reject blobs on the moving hand/grip (d<D_LO) and far balls (d>D_HI).

Validated on video 382 (13 contact / 38 shadow, checked by eye). NOTE: HSV range and
the D_LO/D_HI band are tuned to 382's lighting + camera distance — re-check on other
clips before trusting broadly. Not yet wired into the segmentation pipeline.

Writes a table + annotated frames to inspect_out/contact/.
Run:  source venv/bin/activate && python ball_contact_probe.py
"""
import json, os, sys
import cv2, numpy as np, mediapipe as mp
from analysis.pose import L, _ROTATE_CODES

VIDEO = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                           else "~/Downloads/My_recorded_video_382.MP4")
STROKES = f"inspect_out/{os.path.basename(VIDEO)}.strokes.json"
SIDE = "r"                       # dominant hand
WIN = 5                          # +/- frames around the peak to search
# A real contact puts the ball at the RACKET HEAD, ~1-2 torso-lengths past the
# wrist. Below D_LO the "ball" is on the hand/grip (yellow wristband/grip tape
# that moves with the hand and slips through the motion filter) -> not a strike.
D_LO, D_HI = 0.5, 2.3
OUTDIR = "inspect_out/contact"
os.makedirs(OUTDIR, exist_ok=True)

# Optic-yellow tennis ball in HSV (OpenCV H is 0-179). Fluorescent yellow-green,
# bright and saturated — kept toward yellow (not the greener foliage/court).
YELLOW_LO = np.array([26, 80, 140])
YELLOW_HI = np.array([42, 255, 255])


def yellow_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def moving_balls(masks, i):
    """Blobs that are yellow in frame i but NOT in the neighbours (i-1, i+1).

    A fast in-play ball is yellow HERE and gone from this spot one frame either
    side. Static yellow (foliage, hopper, balls on the ground, a held ball that
    barely moves) is yellow in all three -> cancels out."""
    KILL = cv2.dilate(np.maximum(masks[i - 1], masks[i + 1]),
                      np.ones((7, 7), np.uint8))          # static/neighbour yellow, dilated
    motion = cv2.bitwise_and(masks[i], cv2.bitwise_not(KILL))
    balls = []
    cnts, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 3 or a > 1500:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if a / (np.pi * r * r + 1e-6) < 0.45:            # roundness
            continue
        balls.append((float(x), float(y), float(r)))
    return balls


def wrist_and_scale(res, w, h):
    """Return (wrist_xy, torso_len_px) from a pose result, or (None, None)."""
    if not res.pose_landmarks:
        return None, None
    lm = res.pose_landmarks.landmark

    def px(name):
        p = lm[L[name]]
        return np.array([p.x * w, p.y * h]) if p.visibility >= 0.4 else None

    wrist = px(f"{SIDE}_wrist")
    sh = [px("l_shoulder"), px("r_shoulder")]
    hip = [px("l_hip"), px("r_hip")]
    scale = None
    if all(p is not None for p in sh + hip):
        shoulder_mid = (sh[0] + sh[1]) / 2
        hip_mid = (hip[0] + hip[1]) / 2
        scale = float(np.linalg.norm(shoulder_mid - hip_mid))   # torso length px
    return wrist, scale


def main():
    windows = json.load(open(STROKES))
    peaks = [w["window"]["peak_time"] for w in windows]
    plausible = []
    for w in windows:
        r = w["report"]
        xf, kn, lt = r["peak_x_factor"], r["min_knee_bend"], r["contact_wrist_x_rel"]
        ok = not ((xf is not None and (xf >= 89 or xf < 8)) or kn is None or kn < 90
                  or (lt is not None and lt < -0.03))
        plausible.append(ok)

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rot = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) % 360
    rcode = _ROTATE_CODES.get(rot)

    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1,
                                  min_detection_confidence=0.4)

    print(f"{'t_peak':>7} {'plaus':>5} {'wrist':>5} {'nballs':>6} {'min_d/torso':>11}  label")
    results = []
    save_every = max(1, len(peaks) // 12)   # ~12 annotated frames to eyeball
    for i, t in enumerate(peaks):
        pk = int(round(t * fps))
        # read the whole window contiguously so motion detection has neighbours
        frames, masks, wrists, scales = [], [], [], []
        cap.set(cv2.CAP_PROP_POS_FRAMES, pk - WIN)
        for f in range(pk - WIN, pk + WIN + 1):
            ret, bgr = cap.read()
            if not ret:
                break
            if rcode is not None:
                bgr = cv2.rotate(bgr, rcode)
            h, w = bgr.shape[:2]
            res = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            wrist, scale = wrist_and_scale(res, w, h)
            frames.append(bgr); masks.append(yellow_mask(bgr))
            wrists.append(wrist); scales.append(scale)

        best = {"d": np.inf, "frame": None, "wrist": None, "balls": [], "fidx": None}
        # fallback for annotating shadow swings (no moving ball found): centre frame
        mid = len(frames) // 2
        fallback = (frames[mid], wrists[mid]) if frames and wrists[mid] is not None else (None, None)
        wrist_seen = False
        for j in range(1, len(frames) - 1):                 # interior frames only
            wrist, scale = wrists[j], scales[j]
            if wrist is None or scale is None or scale < 1:
                continue
            wrist_seen = True
            balls = moving_balls(masks, j)
            for (bx, by, br) in balls:
                d = np.hypot(bx - wrist[0], by - wrist[1]) / scale
                if d < D_LO:            # on the moving hand/grip, not a struck ball
                    continue
                if d < best["d"]:
                    best.update(d=d, frame=frames[j].copy(), wrist=wrist,
                                balls=balls, fidx=pk - WIN + j)
        min_d = best["d"] if np.isfinite(best["d"]) else None
        # a moving ball in the racket-head band [D_LO, D_HI] at the swing peak =
        # contact; nothing in the band (absent, on-hand, or far) = shadow swing
        if not wrist_seen:
            label = "?"                       # couldn't locate wrist, can't judge
        elif min_d is None:
            label = "shadow"                  # no moving ball in the racket-head band
        else:
            label = "contact" if min_d <= D_HI else "shadow"
        results.append({"t": t, "plausible": plausible[i], "min_d": min_d, "label": label})
        print(f"{t:>7.1f} {str(plausible[i]):>5} {str(wrist_seen):>5} "
              f"{len(best['balls']):>6} {('%.2f'%min_d) if min_d is not None else '  --':>11}  {label}")

        if i % save_every == 0:
            img = best["frame"] if best["frame"] is not None else (
                fallback[0].copy() if fallback[0] is not None else None)
            wr = best["wrist"] if best["wrist"] is not None else fallback[1]
            if img is not None and wr is not None:
                cv2.circle(img, tuple(wr.astype(int)), 10, (255, 0, 255), 2)
                for (bx, by, br) in best["balls"]:
                    cv2.circle(img, (int(bx), int(by)), max(6, int(br) + 4), (0, 255, 255), 2)
                dtxt = f"d={min_d:.2f}" if min_d is not None else "no-moving-ball"
                cv2.putText(img, f"t={t:.1f} {dtxt} {label}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imwrite(f"{OUTDIR}/swing_{t:.1f}.png", img)
    cap.release()

    # summary over plausible swings only
    ps = [r for r in results if r["plausible"] and r["min_d"] is not None]
    if ps:
        ds = sorted(r["min_d"] for r in ps)
        nc = sum(1 for r in ps if r["label"] == "contact")
        print(f"\nplausible swings with wrist+scale: {len(ps)}")
        print(f"  min wrist-ball dist (torso units): "
              f"min={ds[0]:.2f} median={ds[len(ds)//2]:.2f} max={ds[-1]:.2f}")
        print(f"  contact={nc}  shadow={len(ps)-nc}  (contact band {D_LO}<=d<={D_HI})")
    print(f"\nannotated frames -> {OUTDIR}/  (verify detection by eye)")


if __name__ == "__main__":
    main()
