import cv2
import threading


class CameraThread(threading.Thread):
    def __init__(self, queue_manager, camera_index=0):
        super().__init__(daemon=True)
        self.queue_manager = queue_manager
        self.camera_index = camera_index
        self._stop_event = threading.Event()
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.queue_manager.put_frame(frame)

        self.cap.release()

    def stop(self):
        self._stop_event.set()
