"""API routes for stroke analysis."""
import os
import uuid
import aiofiles
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from analysis.pose import extract_poses
from analysis.oti import compute_oti_metrics
from analysis.coach import generate_feedback

router = APIRouter()

UPLOAD_DIR = "/tmp/stroke-lab-uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/analyze")
async def analyze_stroke(
    video: UploadFile = File(...),
    stroke_type: str = Form(default="forehand"),
    player_side: str = Form(default="r"),
):
    """
    Upload a video and receive OTI analysis + coaching feedback.
    stroke_type: "forehand" | "backhand" | "serve"
    player_side: "r" (right-handed) | "l" (left-handed)
    """
    if not video.filename.lower().endswith((".mp4", ".mov", ".avi", ".m4v")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    video_id = str(uuid.uuid4())
    ext = os.path.splitext(video.filename)[1]
    video_path = os.path.join(UPLOAD_DIR, f"{video_id}{ext}")

    async with aiofiles.open(video_path, "wb") as f:
        content = await video.read()
        await f.write(content)

    try:
        seq = extract_poses(video_path, sample_every_n=2)
        if not seq.frames:
            raise HTTPException(status_code=422, detail="No pose detected in video")

        report = compute_oti_metrics(seq, player_side=player_side)
        result = generate_feedback(report, stroke_type=stroke_type)
        result["video_id"] = video_id
        result["frame_count"] = len(seq.frames)
        result["duration_sec"] = round(seq.frames[-1].time_sec, 1) if seq.frames else 0
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

    return JSONResponse(content=result)


@router.get("/health")
async def health():
    return {"status": "ok"}
