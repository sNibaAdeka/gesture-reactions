from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from domain_types import Candidate, FaceFeatures, HandFeatures, PoseFeatures, ReactionState


class Renderer:
    def __init__(self, assets_dir: Path, overlay_seconds: float, fade_seconds: float) -> None:
        self.overlay_seconds = overlay_seconds
        self.fade_seconds = fade_seconds
        self.assets: dict[ReactionState, np.ndarray] = {}
        self.asset_warning: str | None = None
        self.active_state = ReactionState.NEUTRAL
        self.active_label = "none"
        self.active_confidence = 0.0
        self.activated_at = 0.0
        self.current_candidate = Candidate()
        self.debug = False
        self.show_hud = False
        self.popup_window = "Jaqyn"
        self.popup_visible = False
        for state, filename in ((ReactionState.POSITIVE, "positive.png"), (ReactionState.NEGATIVE, "negative.png")):
            image = cv2.imread(str(assets_dir / filename), cv2.IMREAD_UNCHANGED)
            if image is None:
                self.asset_warning = f"Missing asset: assets/{filename}; using a built-in placeholder."
                image = self._placeholder(state)
            self.assets[state] = image
        self.positive_popup = cv2.imread(str(assets_dir / "positive_popup.png"), cv2.IMREAD_COLOR)

    def reset(self) -> None:
        self.active_state = ReactionState.NEUTRAL
        self.active_label = "none"
        self.active_confidence = 0.0
        self._close_popup()

    def activate(self, state: ReactionState, label: str, confidence: float, now: float) -> None:
        self.active_state, self.active_label, self.active_confidence, self.activated_at = state, label, confidence, now
        if state is ReactionState.POSITIVE:
            self._show_positive_popup()

    def update_detection(self, candidate: Candidate) -> None:
        self.current_candidate = candidate

    def draw(self, frame: np.ndarray, face: FaceFeatures, hand: HandFeatures, pose: PoseFeatures, fps: float, calibrating: bool, calibration_remaining: float) -> np.ndarray:
        output = frame.copy()
        if self.debug:
            self._draw_landmarks(output, face, hand, pose)
        now = time.monotonic()
        if self.active_state is not ReactionState.NEUTRAL:
            age = now - self.activated_at
            if age <= self.overlay_seconds:
                self._overlay(output, self.assets[self.active_state], min(1.0, age / self.fade_seconds))
            else:
                self.reset()
        if self.show_hud:
            self._panel(output, fps, calibrating, calibration_remaining)
        return output

    def _show_positive_popup(self) -> None:
        if self.positive_popup is None:
            return
        image = self.positive_popup
        max_width, max_height = 800, 700
        scale = min(max_width / image.shape[1], max_height / image.shape[0], 1.0)
        shown = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        cv2.namedWindow(self.popup_window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.popup_window, shown)
        self.popup_visible = True

    def _close_popup(self) -> None:
        if self.popup_visible:
            cv2.destroyWindow(self.popup_window)
            self.popup_visible = False

    def _draw_landmarks(self, frame: np.ndarray, face: FaceFeatures, hand: HandFeatures, pose: PoseFeatures) -> None:
        height, width = frame.shape[:2]
        for landmark in face.landmarks[::4]:
            cv2.circle(frame, (int(landmark.x * width), int(landmark.y * height)), 1, (90, 220, 90), -1)
        for points in hand.landmarks:
            for landmark in points:
                cv2.circle(frame, (int(landmark.x * width), int(landmark.y * height)), 3, (255, 180, 30), -1)
        for landmark in pose.landmarks:
            if getattr(landmark, "visibility", 1.0) > 0.4:
                cv2.circle(frame, (int(landmark.x * width), int(landmark.y * height)), 3, (230, 80, 190), -1)

    def _panel(self, frame: np.ndarray, fps: float, calibrating: bool, remaining: float) -> None:
        cv2.rectangle(frame, (10, 10), (390, 135), (12, 12, 12), -1)
        color = {ReactionState.NEUTRAL: (220, 220, 220), ReactionState.POSITIVE: (70, 230, 100), ReactionState.NEGATIVE: (70, 70, 245)}[self.active_state]
        lines = [
            (f"STATUS: {self.active_state.value}", color),
            (f"Found: {self.current_candidate.label}", (220, 220, 220)),
            (f"Confidence: {self.current_candidate.confidence:.2f}    FPS: {fps:.1f}", (220, 220, 220)),
            ("q quit | d debug | r reset | c calibrate", (170, 170, 170)),
        ]
        if calibrating:
            lines[0] = (f"CALIBRATING: look neutral ({remaining:.1f}s)", (40, 210, 255))
        for index, (text, line_color) in enumerate(lines):
            cv2.putText(frame, text, (20, 35 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, line_color, 2, cv2.LINE_AA)
        if self.asset_warning:
            cv2.putText(frame, self.asset_warning, (14, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _placeholder(state: ReactionState) -> np.ndarray:
        image = np.zeros((220, 220, 4), dtype=np.uint8)
        color = (50, 210, 100, 240) if state is ReactionState.POSITIVE else (70, 70, 240, 240)
        image[:, :] = color
        cv2.putText(image, state.value, (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255, 255), 2, cv2.LINE_AA)
        return image

    @staticmethod
    def _overlay(frame: np.ndarray, image: np.ndarray, alpha: float) -> None:
        height, width = frame.shape[:2]
        target_width = min(int(width * 0.30), image.shape[1])
        scale = target_width / image.shape[1]
        target_height = max(1, int(image.shape[0] * scale))
        overlay = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        x, y = width - target_width - 25, 155
        y = min(y, max(0, height - target_height - 10))
        roi = frame[y : y + target_height, x : x + target_width]
        if overlay.shape[2] == 4:
            mask = (overlay[:, :, 3:4].astype(np.float32) / 255.0) * alpha
            roi[:] = (roi * (1.0 - mask) + overlay[:, :, :3] * mask).astype(np.uint8)
        else:
            cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
