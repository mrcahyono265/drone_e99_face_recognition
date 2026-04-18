import cv2
import time
import threading
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"

class cameraDroneThread:
    def __init__(self, rtsp_url):
        print("[INFO]: Connecting drone...")
        self.stream = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # read first frame
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Thread starting
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        # Looping
        while True:
            if self.stopped:
                self.stream.release()
                return
            # read next frame
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        # Call looping 
        if self.frame is not None:
            return cv2.rotate(self.frame, cv2.ROTATE_90_CLOCKWISE)
        return None

    def stop(self):
        self.stopped = True
