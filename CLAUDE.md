# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## The goal (read this first)

StrokeLab is meant to be **HomeCourt for tennis**: while you practice — starting with the forehand — you get **realtime coaching cues in your AirPods, after each stroke**, so you can tell whether you're improving without stopping to review video. The product value is the tight in-ear feedback loop during a live practice session, and tracking progress rep-over-rep. It is *not* a post-session report generator.

The north star flow is: **live video stream → segment individual strokes as they happen → score each stroke (OTI) → speak a short cue in the player's ear → track trend across reps.**

**The current code is an offline stepping stone, wired the wrong way for that goal.** Weigh proposals against the north star, not against the existing endpoint:
- **Batch, not realtime.** `POST /api/analyze` ingests one whole uploaded file, runs MediaPipe end-to-end, then makes a multi-second Claude call. That's a review tool; the goal needs low-latency, per-stroke turnaround.
- **Text, not audio.** Output is a JSON feedback blob. There is no TTS / audio-cue path yet, which is the actual delivery mechanism.
- **One stroke per upload.** Real practice is a continuous stream of strokes. `analysis/segment.py` (per-stroke splitting) is therefore *on the critical path*, not a side feature — "feedback after every swing" is impossible without it. See the segmentation note below.

So `analysis/pose.py`, `analysis/oti.py`, and `analysis/coach.py` are the right building blocks (pose → metrics → cue), but everything is currently assembled for offline single-clip review rather than a live per-stroke loop.

## What it does today

The OTI (Optimum Tennis Instruction) methodology drives the metrics: body pose from MediaPipe → biomechanical metrics for three power sources (legs, shoulders, late hit) → Claude turns metrics into coaching feedback. Today that runs as a batch upload-and-return.

Monorepo with two independent halves:
- `backend/` — Python 3.11 · FastAPI · MediaPipe · OpenCV · Anthropic
- `mobile/` — React Native · Expo (SDK 56) · TypeScript

## Commands

### Backend
```bash
cd backend
source venv/bin/activate        # venv is committed-ignored but expected to exist locally
python main.py                  # serves FastAPI on http://0.0.0.0:8000 (reload on)
export ANTHROPIC_API_KEY=sk-ant-...   # required — coach.py instantiates anthropic.Anthropic() at import
```

There is **no test suite**. Validation is done with standalone inspection scripts that run the pipeline on real footage and print raw signals (they do **not** call Claude, so no API key needed). Their output goes to the git-ignored `backend/inspect_out/`; sample videos live in git-ignored `backend/samples/`:
```bash
python inspect_clip.py samples/clip1.mov --side r --stroke forehand   # single-stroke pipeline dump
python recon_scan.py path/to/session.MP4                              # fast recon: detection coverage + metric timeline
EVERY_N=5 python recon_scan.py path/to/session.MP4                    # denser sampling
python segment_run.py path/to/session.MP4 --side r --every 2 --complexity 1  # multi-stroke segmentation + per-stroke metrics
```

### Mobile
```bash
cd mobile
npx expo start                  # then scan the QR in Expo Go
```
When testing on a physical device, set `BASE_URL` in `mobile/src/api/client.ts` to your machine's LAN IP (e.g. `http://192.168.1.x:8000/api`) — `localhost` won't reach the backend from the phone.

`mobile/AGENTS.md` (loaded by `mobile/CLAUDE.md`) requires reading the exact versioned Expo docs at `https://docs.expo.dev/versions/v56.0.0/` before writing any Expo code — the SDK is pinned and APIs have changed.

## Architecture

### Backend pipeline
The served path is a three-stage pipeline, orchestrated by `POST /api/analyze` in `api/routes.py`:

1. **`analysis/pose.py` — `extract_poses()`** decodes the video with OpenCV and runs MediaPipe Pose per frame, producing a `PoseSequence` of `Frame`s. Each `Frame` carries **two coordinate systems** that the rest of the code selects between deliberately:
   - `landmarks` / `frame.pt(name)` → 2D **pixel** coords (normalized × width/height). Used for in-image geometry (knee angle, wrist-vs-hip).
   - `world_landmarks` / `frame.world(name)` → 3D **metric, hip-centered** coords (x = subject's right, y = up, z = toward camera). Used where a real 3D angle is needed (X-factor).
   - Both return `None` when landmark visibility < 0.5; every consumer must handle `None`.

2. **`analysis/oti.py` — `compute_oti_metrics()`** turns the pose sequence into an `OTIReport`: peak X-factor (3D shoulder-vs-hip yaw separation), min knee bend, contact point (wrist-x relative to front hip), and 0–10 scores per power source. Scores use hardcoded piecewise-linear thresholds.

3. **`analysis/coach.py` — `generate_feedback()`** formats the report into a prompt and calls Claude (model `claude-opus-4-7`, hardcoded) with the OTI system prompt, returning the metrics + scores + natural-language feedback as the API response.

`main.py` is a thin FastAPI app (CORS `*`, router mounted under `/api`). Uploads are written to `/tmp/stroke-lab-uploads` and deleted after analysis.

### Segmentation is not wired into the API
`analysis/segment.py` (`find_stroke_windows`) splits a long multi-stroke session into per-stroke windows using dominant-wrist-speed peaks with non-max suppression, and filters out non-strokes via torso uprightness. **It is only invoked by the `segment_run.py` CLI, not by `/api/analyze`** — the served endpoint still assumes one stroke per upload. Real-world clips are long, rotated, portrait, multi-stroke sessions, which is what segmentation exists to handle. Segmentation currently produces false positives.

This gap is the main thing standing between today's code and the north star: per-stroke detection is what makes "a cue after every swing" possible, so a live/streaming version of `find_stroke_windows` (detecting a completed stroke from a rolling buffer, low-latency, offline batch removed) is closer to the real product than anything on the `/analyze` path.

### Conventions and gotchas that span files
- **Handedness / "front hip":** `player_side` is `"r"` (right-handed) or `"l"` (lefty). The *front* hip (nearest the net) is the **opposite** side from the dominant hand for a forehand. This mapping recurs in `oti.py` and the CLI scripts.
- **Landmark naming** (`l_`/`r_` in `pose.L`) is anatomical, from the subject's own perspective — not screen-left/right.
- **Rotation metadata is applied manually** in `extract_poses` (`_ROTATE_CODES`) because OpenCV's auto-orientation is a no-op on many builds. Phone videos store a rotation flag; skipping this analyzes a sideways player and every angle becomes garbage. Width/height are swapped for 90°/270°.
- **Peak X-factor uses the 95th percentile, not `max()`** (`oti.py`), to reject spurious high values from bad-detection frames. Prefer robust statistics (percentile/median/MAD) over raw extrema throughout the metrics code — bad pose frames are common.
- `swing_direction` is currently stubbed to `"neutral"` (`_wrist_trajectory_direction`); it needs real 2D wrist-path analysis from rear-view footage.

### Backend ↔ mobile contract
The mobile app posts multipart form-data (`video`, `stroke_type`, `player_side`) to `/api/analyze` from `mobile/src/api/client.ts`. The `AnalysisResult` TypeScript interface there must stay in sync with the dict assembled in `coach.py` + `routes.py` (`scores`, `metrics`, `feedback`, `swing_direction`, plus `video_id`/`frame_count`/`duration_sec`).
