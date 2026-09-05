from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    url: str


@dataclass
class AppConfig:
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    analysis_width: int = 640
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.55
    calibration_seconds: float = 3.0
    temporal_window: int = 12
    majority_ratio: float = 0.67
    min_gesture_seconds: float = 0.40
    cooldown_seconds: float = 1.0
    overlay_seconds: float = 2.5
    overlay_fade_seconds: float = 0.25
    eye_closed_ratio: float = 0.62
    fallback_eye_ear: float = 0.23
    mouth_open_ratio: float = 1.75
    fallback_mouth_opening: float = 0.070
    tongue_min_score: float = 0.09
    thumb_extended_angle: float = 145.0
    finger_folded_angle: float = 145.0
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    assets_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "assets")

    @property
    def model_specs(self) -> tuple[ModelSpec, ...]:
        return (
            ModelSpec(
                "face_landmarker.task",
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            ),
            ModelSpec(
                "hand_landmarker.task",
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            ),
            ModelSpec(
                "pose_landmarker_lite.task",
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            ),
        )
