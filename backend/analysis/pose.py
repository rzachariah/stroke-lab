"""Extract MediaPipe pose landmarks from video frames."""
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


mp_pose = mp.solutions.pose

# Landmark indices we care about
L = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
}


@dataclass
class Frame:
    index: int
    time_sec: float
    landmarks: Optional[dict] = None  # name -> (x, y, z, visibility)
    width: int = 0
    height: int = 0

    def pt(self, name: str) -> Optional[np.ndarray]:
        """Return (x, y) pixel coords for a landmark, or None if low visibility."""
        if self.landmarks is None:
            return None
        lm = self.landmarks.get(name)
        if lm is None or lm[3] < 0.5:
            return None
        return np.array([lm[0] * self.width, lm[1] * self.height])

    def pt3(self, name: str) -> Optional[np.ndarray]:
        """Return (x, y, z) normalized coords."""
        if self.landmarks is None:
            return None
        lm = self.landmarks.get(name)
        if lm is None or lm[3] < 0.5:
            return None
        return np.array([lm[0], lm[1], lm[2]])


@dataclass
class PoseSequence:
    frames: list[Frame] = field(default_factory=list)
    fps: float = 30.0


def extract_poses(video_path: str, sample_every_n: int = 1) -> PoseSequence:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    seq = PoseSequence(fps=fps)
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                landmarks = None
                if results.pose_landmarks:
                    landmarks = {}
                    for name, idx in L.items():
                        lm = results.pose_landmarks.landmark[idx]
                        landmarks[name] = (lm.x, lm.y, lm.z, lm.visibility)

                seq.frames.append(Frame(
                    index=frame_idx,
                    time_sec=frame_idx / fps,
                    landmarks=landmarks,
                    width=width,
                    height=height,
                ))

            frame_idx += 1

    cap.release()
    return seq
