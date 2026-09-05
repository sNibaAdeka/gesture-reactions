from __future__ import annotations

import math

import cv2
import mediapipe as mp
import numpy as np

from detectors.tongue_detector import TongueDetector
from domain_types import FaceFeatures


def _distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _ear(points: list, outer: int, top_a: int, bottom_a: int, top_b: int, bottom_b: int, inner: int) -> float:
    horizontal = _distance(points[outer], points[inner])
    return (_distance(points[top_a], points[bottom_a]) + _distance(points[top_b], points[bottom_b])) / max(2 * horizontal, 1e-6)


class FaceDetector:
    def __init__(self, model_path: str, min_confidence: float, min_tracking: float) -> None:
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_tracking,
            output_face_blendshapes=True,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.tongue = TongueDetector()

    def detect(self, rgb_frame: np.ndarray, bgr_frame: np.ndarray, timestamp_ms: int) -> FaceFeatures:
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame), timestamp_ms)
        if not result.face_landmarks:
            return FaceFeatures()
        landmarks = list(result.face_landmarks[0])
        left = _ear(landmarks, 33, 160, 144, 158, 153, 133)
        right = _ear(landmarks, 362, 385, 380, 387, 373, 263)
        mouth = _distance(landmarks[13], landmarks[14]) / max(_distance(landmarks[78], landmarks[308]), 1e-6)
        face_size = _distance(landmarks[234], landmarks[454])
        blendshapes = {}
        if result.face_blendshapes:
            blendshapes = {item.category_name: float(item.score) for item in result.face_blendshapes[0]}
        return FaceFeatures(
            landmarks=landmarks,
            eye_ear=(left + right) / 2,
            mouth_opening=mouth,
            face_size=face_size,
            blendshapes=blendshapes,
            tongue_score=self.tongue.score(bgr_frame, landmarks, mouth),
        )

    def close(self) -> None:
        self.landmarker.close()
