import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SRC_DIR)

import cv2

from src.live.pipeline import build_pipeline
from src.utils.visualization import draw_emotion_overlay, FPSCounter

CONFIG_PATH = os.path.join(ROOT_DIR, 'config.yaml')


def main():
    cam_thread, inference_thread, queue_manager, cfg = build_pipeline(CONFIG_PATH)

    cam_thread.start()
    inference_thread.start()

    fps_counter = FPSCounter()
    display_fps = cfg['live']['display_fps']

    last_label = 'detecting...'
    last_confidence = 0.0
    last_face_rect = None

    print("Live emotion recognition started. Press ESC to quit.")

    while True:
        frame = queue_manager.get_frame(timeout=0.05)
        if frame is None:
            continue

        result = queue_manager.get_result()
        if result is not None:
            last_label, last_confidence, last_face_rect = result

        fps_counter.update()
        fps = fps_counter.fps if display_fps else None

        annotated = draw_emotion_overlay(
            frame.copy(),
            last_label,
            last_confidence,
            fps=fps,
            face_rect=last_face_rect,
        )

        cv2.imshow('Emotion Recognition', annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cam_thread.stop()
    inference_thread.stop()
    cam_thread.join()
    inference_thread.join()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == '__main__':
    main()
