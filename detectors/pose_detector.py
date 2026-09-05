from __future__ import annotations

import mediapipe as mp
import numpy as np

from domain_types import PoseFeatures


class PoseDetector:
    def __init__(self, model_path: str, min_confidence: float, min_tracking: float) -> None:
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_tracking,
            output_segmentation_masks=False,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> PoseFeatures:
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame), timestamp_ms)
        return PoseFeatures(list(result.pose_landmarks[0]) if result.pose_landmarks else [])

    def close(self) -> None:
        self.landmarker.close()
