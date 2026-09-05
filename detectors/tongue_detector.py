from __future__ import annotations

import cv2
import numpy as np


class TongueDetector:
    """A replaceable colour heuristic; Face Landmarker itself has no tongue landmarks."""

    def score(self, bgr_frame: np.ndarray, landmarks: list, mouth_opening: float | None) -> float:
        if not landmarks or mouth_opening is None or mouth_opening < 0.045:
            return 0.0
        height, width = bgr_frame.shape[:2]
        points = np.array(
            [[landmarks[i].x * width, landmarks[i].y * height] for i in (78, 308, 13, 14)], dtype=np.int32
        )
        x, y, box_width, box_height = cv2.boundingRect(points)
        pad_x, pad_y = int(box_width * 0.18), int(box_height * 0.45 + 3)
        x0, x1 = max(0, x - pad_x), min(width, x + box_width + pad_x)
        y0, y1 = max(0, y - pad_y), min(height, y + box_height + pad_y)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        roi = bgr_frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_red = cv2.inRange(hsv, (0, 55, 45), (18, 255, 255))
        upper_red = cv2.inRange(hsv, (165, 55, 45), (179, 255, 255))
        # Prefer the lower inner mouth: lips at the border should contribute less.
        mask = cv2.bitwise_or(lower_red, upper_red)
        inner = mask[mask.shape[0] // 3 :, mask.shape[1] // 5 : -mask.shape[1] // 5 or None]
        return float(np.count_nonzero(inner) / max(1, inner.size))
