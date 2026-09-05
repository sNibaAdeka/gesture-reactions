from __future__ import annotations

import time

import cv2

from camera import Camera
from config import AppConfig
from detectors.face_detector import FaceDetector
from detectors.hand_detector import HandDetector
from detectors.pose_detector import PoseDetector
from domain_types import Candidate
from logic.calibration import Calibrator
from logic.expression_classifier import ExpressionClassifier
from logic.gesture_classifier import GestureClassifier
from logic.temporal_filter import TemporalFilter
from models import ensure_models
from ui.renderer import Renderer


def _choose_candidate(hand_candidate: Candidate, expression_candidate: Candidate) -> Candidate:
    if expression_candidate.confidence >= hand_candidate.confidence:
        return expression_candidate
    return hand_candidate


def main() -> None:
    config = AppConfig()
    model_paths = ensure_models(config.models_dir, config.model_specs)
    camera = Camera(config.camera_index, config.frame_width, config.frame_height)
    face_detector = hand_detector = pose_detector = None
    try:
        face_detector = FaceDetector(str(model_paths["face_landmarker.task"]), config.min_detection_confidence, config.min_tracking_confidence)
        hand_detector = HandDetector(str(model_paths["hand_landmarker.task"]), config.min_detection_confidence, config.min_tracking_confidence, config.thumb_extended_angle, config.finger_folded_angle)
        pose_detector = PoseDetector(str(model_paths["pose_landmarker_lite.task"]), config.min_detection_confidence, config.min_tracking_confidence)
        gesture_classifier = GestureClassifier()
        expression_classifier = ExpressionClassifier(config)
        temporal_filter = TemporalFilter(config.temporal_window, config.majority_ratio, config.min_gesture_seconds, config.cooldown_seconds)
        calibrator = Calibrator(config.calibration_seconds)
        renderer = Renderer(config.assets_dir, config.overlay_seconds, config.overlay_fade_seconds)
        calibrator.start(time.monotonic())
        fps, last_frame_at, last_timestamp = 0.0, time.monotonic(), 0
        cv2.namedWindow("Gesture Reactions", cv2.WINDOW_NORMAL)

        while True:
            frame = camera.read()
            if frame.shape[1] > config.analysis_width:
                scale = config.analysis_width / frame.shape[1]
                frame = cv2.resize(frame, (config.analysis_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            now = time.monotonic()
            timestamp_ms = max(last_timestamp + 1, int(now * 1000))
            last_timestamp = timestamp_ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = face_detector.detect(rgb, frame, timestamp_ms)
            hand = hand_detector.detect(rgb, timestamp_ms)
            pose = pose_detector.detect(rgb, timestamp_ms)

            if calibrator.update(face, now):
                print("Calibration complete.")
            hand_candidate = gesture_classifier.classify(hand)
            expression_candidate, _ = expression_classifier.classify(face, calibrator.baseline, now)
            candidate = _choose_candidate(hand_candidate, expression_candidate)
            renderer.update_detection(candidate)
            if not calibrator.active:
                event = temporal_filter.update(candidate, now)
                if event is not None:
                    renderer.activate(event.state, event.label, event.confidence, now)

            elapsed = max(now - last_frame_at, 1e-6)
            fps = 0.90 * fps + 0.10 * (1.0 / elapsed) if fps else 1.0 / elapsed
            last_frame_at = now
            output = renderer.draw(frame, face, hand, pose, fps, calibrator.active, calibrator.remaining(now))
            cv2.imshow("Gesture Reactions", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                renderer.debug = not renderer.debug
            if key == ord("r"):
                renderer.reset()
                temporal_filter.reset()
            if key == ord("c"):
                calibrator.start(now)
                temporal_filter.reset()
                renderer.reset()
    finally:
        for detector in (face_detector, hand_detector, pose_detector):
            if detector is not None:
                detector.close()
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
