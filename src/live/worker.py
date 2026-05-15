import threading
import cv2
import numpy as np
from PIL import Image


class InferenceWorker(threading.Thread):
    def __init__(self, queue_manager, model, transform, classes, device,
                 face_scale_factor=1.1, face_min_neighbors=5):
        super().__init__(daemon=True)
        self.queue_manager = queue_manager
        self.model = model
        self.transform = transform
        self.classes = classes
        self.device = device
        self.face_scale_factor = face_scale_factor
        self.face_min_neighbors = face_min_neighbors
        self._stop_event = threading.Event()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def _detect_face(self, gray_frame):
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=self.face_scale_factor,
            minNeighbors=self.face_min_neighbors,
            minSize=(30, 30),
        )
        if len(faces) == 0:
            return None
        # return largest face
        areas = [w * h for (x, y, w, h) in faces]
        return faces[np.argmax(areas)]

    def _run_inference(self, face_roi):
        from src.inference import predict_image
        pil_img = Image.fromarray(face_roi)
        label, confidence, _ = predict_image(pil_img, self.model, self.transform, self.classes, self.device)
        return label, confidence

    def run(self):
        while not self._stop_event.is_set():
            frame = self.queue_manager.get_frame(timeout=0.1)
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rect = self._detect_face(gray)

            if face_rect is not None:
                x, y, w, h = face_rect
                face_roi = gray[y:y + h, x:x + w]
                label, confidence = self._run_inference(face_roi)
                self.queue_manager.put_result((label, confidence, face_rect))

    def stop(self):
        self._stop_event.set()
