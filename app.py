import cv2
import time
import os
from datetime import datetime

class UI:
    def __init__(self, output_dir="inference_output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.is_recording = False
        self.out_video = None
        self.frame_h = 0
        self.frame_w = 0

    def take_snapshot(self, frame):
        """Excecute when 'S' key is pressed for taking snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame) 
        print(f"[INFO] Snapshot taken and saved: {filename}")

    def toggle_recording(self):
        """Execute when 'R' key is pressed for toggling recording"""
        if not self.is_recording:
            # START RECORDING
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/video_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out_video = cv2.VideoWriter(filename, fourcc, 25.0, (self.frame_w, self.frame_h))
            self.is_recording = True
            print(f"[INFO] Start recording: {filename}")
        else:
            # END RECORDING
            self.is_recording = False
            if self.out_video is not None:
                self.out_video.release()
                self.out_video = None
            print("[INFO] Recording ended")

    def process_ui(self, frame):
        # Process UI elements for each frame
        self.frame_h, self.frame_w = frame.shape[:2]

        # Write video if recording
        if self.is_recording and self.out_video is not None:
            self.out_video.write(frame)

        # Show recording indicator
        if self.is_recording and int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (self.frame_w - 20, 25), 6, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (self.frame_w - 65, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    def cleanup(self):
        if self.out_video is not None:
            self.out_video.release()