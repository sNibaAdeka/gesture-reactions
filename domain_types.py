from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReactionState(str, Enum):
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


@dataclass
class Candidate:
    state: ReactionState = ReactionState.NEUTRAL
    label: str = "none"
    confidence: float = 0.0


@dataclass
class FaceFeatures:
    landmarks: list[Any] = field(default_factory=list)
    eye_ear: float | None = None
    mouth_opening: float | None = None
    face_size: float | None = None
    blendshapes: dict[str, float] = field(default_factory=dict)
    tongue_score: float = 0.0


@dataclass
class HandFeatures:
    landmarks: list[list[Any]] = field(default_factory=list)
    handedness: list[str] = field(default_factory=list)
    gesture: str = "none"
    confidence: float = 0.0


@dataclass
class PoseFeatures:
    landmarks: list[Any] = field(default_factory=list)


@dataclass
class CalibrationBaseline:
    eye_ear: float | None = None
    mouth_opening: float | None = None
    face_size: float | None = None
