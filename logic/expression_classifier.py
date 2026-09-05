from __future__ import annotations

from dataclasses import dataclass

from config import AppConfig
from domain_types import CalibrationBaseline, Candidate, FaceFeatures, ReactionState


@dataclass
class ExpressionDetails:
    eyes_closed: bool = False
    mouth_open: bool = False


class ExpressionClassifier:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.eyes_closed_since: float | None = None

    def classify(self, face: FaceFeatures, baseline: CalibrationBaseline, now: float) -> tuple[Candidate, ExpressionDetails]:
        if face.eye_ear is None or face.mouth_opening is None:
            self.eyes_closed_since = None
            return Candidate(), ExpressionDetails()
        eye_baseline = baseline.eye_ear or self.config.fallback_eye_ear
        mouth_baseline = baseline.mouth_opening or self.config.fallback_mouth_opening
        eyes_closed = face.eye_ear < eye_baseline * self.config.eye_closed_ratio
        mouth_open = face.mouth_opening > max(self.config.fallback_mouth_opening, mouth_baseline * self.config.mouth_open_ratio)
        if eyes_closed:
            self.eyes_closed_since = self.eyes_closed_since or now
        else:
            self.eyes_closed_since = None
        closed_for = now - self.eyes_closed_since if self.eyes_closed_since is not None else 0.0
        details = ExpressionDetails(eyes_closed=eyes_closed, mouth_open=mouth_open)
        if mouth_open and face.tongue_score >= self.config.tongue_min_score:
            confidence = min(1.0, 0.62 + face.tongue_score * 1.8)
            return Candidate(ReactionState.NEGATIVE, "tongue heuristic", confidence), details
        if mouth_open and closed_for >= 0.45:
            return Candidate(ReactionState.NEGATIVE, "open mouth + eyes squeezed", 0.86), details
        if closed_for >= 0.50:
            return Candidate(ReactionState.NEGATIVE, "eyes squeezed", min(0.90, 0.65 + closed_for * 0.2)), details
        frown = (face.blendshapes.get("mouthFrownLeft", 0.0) + face.blendshapes.get("mouthFrownRight", 0.0)) / 2
        blink = (face.blendshapes.get("eyeBlinkLeft", 0.0) + face.blendshapes.get("eyeBlinkRight", 0.0)) / 2
        # This is intentionally labelled as a visual cue, not as emotional inference.
        if frown > 0.55 and blink > 0.38:
            return Candidate(ReactionState.NEGATIVE, "cry-like facial cue", min(0.9, (frown + blink) / 2)), details
        return Candidate(), details
