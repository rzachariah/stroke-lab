"""Generate OTI coaching feedback via Claude."""
import anthropic
from .oti import OTIReport

_client = anthropic.Anthropic()

_SYSTEM = """You are an expert tennis coach trained in the OTI (Optimum Tennis Instruction) methodology.
OTI identifies three primary power sources:
  1. Legs — deep knee bend in preparation, explosive drive at contact
  2. Shoulders — maximum X-factor (shoulder-hip separation) on backswing, full rotation through contact
  3. Late hitting — contact point well in front of the front hip for power and racquet-head acceleration

You also look at kinetic chain activation (legs → hips → trunk → shoulder → arm → racquet) and
swing path (inside-out preferred for topspin and power).

Give specific, actionable feedback. Be direct. Maximum 4 bullet points per section. Use plain text."""


def generate_feedback(report: OTIReport, stroke_type: str = "forehand") -> dict:
    metrics_text = f"""
Stroke type: {stroke_type}

OTI Metrics:
- Peak X-factor (shoulder-hip separation): {report.peak_x_factor}° at t={report.peak_x_factor_time}s
- Minimum knee bend angle: {report.min_knee_bend}° (180°=straight, goal is <145°)
- Contact point (wrist ahead of front hip): {report.contact_wrist_x_rel} (positive = in front, goal >0.05)
- Swing direction: {report.swing_direction}

Scores (0-10):
- Leg power: {report.leg_score}/10
- Shoulder/X-factor: {report.shoulder_score}/10
- Late hit: {report.late_hit_score}/10
"""

    message = _client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Analyze these OTI metrics and give me specific coaching feedback:\n{metrics_text}"
            }
        ]
    )

    feedback_text = message.content[0].text

    return {
        "metrics": report.to_dict(),
        "scores": {
            "legs": report.leg_score,
            "shoulders": report.shoulder_score,
            "late_hit": report.late_hit_score,
            "overall": round(
                ((report.leg_score or 0) + (report.shoulder_score or 0) + (report.late_hit_score or 0)) / 3, 1
            ),
        },
        "feedback": feedback_text,
        "swing_direction": report.swing_direction,
    }
