import cv2
import time


class FPSCounter:
    def __init__(self):
        self._prev = time.time()
        self.fps = 0.0

    def update(self):
        now = time.time()
        self.fps = 1.0 / max(now - self._prev, 1e-6)
        self._prev = now


def draw_emotion_overlay(frame, label, confidence, fps=None, face_rect=None):
    h, w = frame.shape[:2]

    if face_rect is not None:
        x, y, fw, fh = face_rect
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        text = f"{label} {confidence * 100:.1f}%"
        cv2.putText(frame, text, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        text = f"{label} {confidence * 100:.1f}%"
        cv2.putText(frame, text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return frame
