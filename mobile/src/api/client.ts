const BASE_URL = "http://localhost:8000/api";

export interface AnalysisResult {
  video_id: string;
  frame_count: number;
  duration_sec: number;
  scores: {
    legs: number;
    shoulders: number;
    late_hit: number;
    overall: number;
  };
  metrics: {
    peak_x_factor: number | null;
    peak_x_factor_time: number | null;
    min_knee_bend: number | null;
    contact_wrist_x_rel: number | null;
    swing_direction: string | null;
  };
  feedback: string;
  swing_direction: string;
}

export async function analyzeVideo(
  videoUri: string,
  strokeType: "forehand" | "backhand" | "serve",
  playerSide: "r" | "l"
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("video", {
    uri: videoUri,
    name: "stroke.mov",
    type: "video/quicktime",
  } as unknown as Blob);
  form.append("stroke_type", strokeType);
  form.append("player_side", playerSide);

  const res = await fetch(`${BASE_URL}/analyze`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${res.status}`);
  }

  return res.json();
}
