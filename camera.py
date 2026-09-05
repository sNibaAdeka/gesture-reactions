from __future__ import annotations

import platform

import cv2


class Camera:
    def __init__(self, index: int, width: int, height: int) -> None:
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(index, backend)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(
                "Camera could not be opened. Check that no other app is using it and grant "
                "Camera access to Terminal/Python in macOS System Settings > Privacy & Security."
            )

    def read(self):
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("A frame could not be read from the camera.")
        return frame

    def close(self) -> None:
        self.capture.release()
