from queue import Queue, Full, Empty


class QueueManager:
    def __init__(self, maxsize=2):
        self.frame_queue = Queue(maxsize=maxsize)
        self.result_queue = Queue(maxsize=maxsize)

    def put_frame(self, frame):
        try:
            self.frame_queue.put_nowait(frame)
        except Full:
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass
            self.frame_queue.put_nowait(frame)

    def get_frame(self, timeout=0.1):
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def put_result(self, result):
        try:
            self.result_queue.put_nowait(result)
        except Full:
            try:
                self.result_queue.get_nowait()
            except Empty:
                pass
            self.result_queue.put_nowait(result)

    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
