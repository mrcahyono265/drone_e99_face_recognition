import cv2
import time
from datetime import datetime
from camera_config import cameraDroneThread
from model import Models

def main():
    # URL RTSP drone camera
    RTSP_URL = "rtsp://192.168.1.1:7070/webcam"
    
    # Initialize and start
    camera = cameraDroneThread(RTSP_URL).start()
    models = Models()

    # 2. Configure Performance Variables
    frame_skip_rate = 4
    frame_count = 0
    fps_smoothed = 0
    prev_time = time.time()
    
    # Saving Temporary face
    last_bounding_boxes = []

    print("[INFO] Running...")

    while True:
        frame = camera.read()
        
        if frame is None:
            continue

        if frame_count % (frame_skip_rate + 1) == 0:
            faces = models.detect_faces(frame)
            
            # Reset previous faces
            last_bounding_boxes = []
            
            # Take note of detected faces for drawing
            for face in faces:
                bbox = face.bbox.astype(int)
                last_bounding_boxes.append(bbox)

        # Make Bounding Box and lable
        for bbox in last_bounding_boxes:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, "Face Detected", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Count FPS
        time_diff = time.time() - prev_time
        if time_diff > 0:
            fps_smoothed = (fps_smoothed * 0.9) + ((1 / time_diff) * 0.1) if fps_smoothed > 0 else (1 / time_diff)
        prev_time = time.time()
        
        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps_smoothed)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show App
        cv2.imshow("Drone E99 Face Recognition and Anti Spoofing", frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()