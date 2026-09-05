from __future__ import annotations

import statistics
import time

from domain_types import CalibrationBaseline, FaceFeatures


class Calibrator:
    def __init__(self, duration_seconds: float) -> None:
        self.duration_seconds = duration_seconds
        self.baseline = CalibrationBaseline()
        self.started_at = 0.0
        self.samples: list[FaceFeatures] = []
        self.active = False

    def start(self, now: float) -> None:
        self.started_at = now
        self.samples = []
        self.active = True

    def update(self, face: FaceFeatures, now: float) -> bool:
        if not self.active:
            return False
        if face.eye_ear is not None and face.mouth_opening is not None and face.face_size is not None:
            self.samples.append(face)
        if now - self.started_at < self.duration_seconds:
            return False
        self.active = False
        if self.samples:
            self.baseline = CalibrationBaseline(
                eye_ear=statistics.median(sample.eye_ear for sample in self.samples if sample.eye_ear is not None),
                mouth_opening=statistics.median(sample.mouth_opening for sample in self.samples if sample.mouth_opening is not None),
                face_size=statistics.median(sample.face_size for sample in self.samples if sample.face_size is not None),
            )
        return True

    def remaining(self, now: float) -> float:
        return max(0.0, self.duration_seconds - (now - self.started_at)) if self.active else 0.0
