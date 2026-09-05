from __future__ import annotations

import math

import mediapipe as mp
import numpy as np

from domain_types import HandFeatures


def _angle(a, b, c) -> float:
    first = np.array([a.x - b.x, a.y - b.y])
    second = np.array([c.x - b.x, c.y - b.y])
    denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-6)
    return math.degrees(math.acos(float(np.clip(np.dot(first, second) / denominator, -1, 1))))


class HandDetector:
    def __init__(self, model_path: str, min_confidence: float, min_tracking: float, thumb_extended_angle: float, finger_folded_angle: float) -> None:
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_tracking,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.thumb_extended_angle = thumb_extended_angle
        self.finger_folded_angle = finger_folded_angle

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> HandFeatures:
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame), timestamp_ms)
        hands = [list(hand) for hand in result.hand_landmarks]
        handedness = [categories[0].category_name for categories in result.handedness] if result.handedness else []
        best_gesture, best_confidence = "none", 0.0
        for hand in hands:
            gesture, confidence = self._classify_thumb(hand)
            if confidence > best_confidence:
                best_gesture, best_confidence = gesture, confidence
        return HandFeatures(hands, handedness, best_gesture, best_confidence)

    def _classify_thumb(self, hand: list) -> tuple[str, float]:
        thumb_angle = _angle(hand[2], hand[3], hand[4])
        folded_angles = [_angle(hand[mcp], hand[pip], hand[tip]) for mcp, pip, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))]
        folded_fraction = sum(angle < self.finger_folded_angle for angle in folded_angles) / 4
        thumb_extended = max(0.0, min(1.0, (thumb_angle - self.thumb_extended_angle) / 25.0))
        # A vertical thumb must also be clearly beyond the wrist; this rejects pointing gestures.
        palm = max(math.hypot(hand[0].x - hand[9].x, hand[0].y - hand[9].y), 1e-6)
        vertical = (hand[0].y - hand[4].y) / palm
        direction = abs(vertical)
        confidence = max(0.0, min(1.0, 0.45 * thumb_extended + 0.35 * folded_fraction + 0.20 * min(direction, 1.0)))
        if thumb_extended < 0.25 or folded_fraction < 0.50 or direction < 0.32:
            return self._classify_index_up(hand, folded_angles)
        return ("thumbs_up" if vertical > 0 else "thumbs_down"), confidence

    def _classify_index_up(self, hand: list, folded_angles: list[float]) -> tuple[str, float]:
        """One raised index finger, with the remaining fingers folded."""
        index_straight = max(0.0, min(1.0, (folded_angles[0] - self.thumb_extended_angle) / 25.0))
        other_folded = sum(angle < self.finger_folded_angle for angle in folded_angles[1:]) / 3
        palm = max(math.hypot(hand[0].x - hand[9].x, hand[0].y - hand[9].y), 1e-6)
        index_height = (hand[0].y - hand[8].y) / palm
        confidence = max(0.0, min(1.0, 0.45 * index_straight + 0.35 * other_folded + 0.20 * min(max(index_height, 0.0), 1.0)))
        if index_straight < 0.30 or other_folded < 0.67 or index_height < 0.35:
            return "none", 0.0
        return "index_up", confidence

    def close(self) -> None:
        self.landmarker.close()
