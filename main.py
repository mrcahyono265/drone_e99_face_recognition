import cv2
import time
import pickle
from matplotlib import text
import numpy as np
import argparse
from datetime import datetime
from webcam_config import cameraDroneThread
from model import Models
from app import UI
from MiniFASNetV2 import MiniFASNetV2

def main():

    # URL RTSP drone camera
    # RTSP_URL = "rtsp://192.168.1.1:7070/webcam"
    CAMERA_INDEX = 1
    
    # Initialize and start
    # camera = cameraDroneThread(RTSP_URL).start()
    camera = cameraDroneThread(CAMERA_INDEX).start()
    models = Models()

    # Call liveness module
    liveness = MiniFASNetV2()

    # Load Database
    try:
        with open("face_db.pkl", "rb") as f:
            print("[INFO] Loading face database...")
            face_db = pickle.load(f)
        print(f"[INFO] Loaded {len(face_db)} faces from database.")
    except Exception as e:
        print(f"[ERROR] Failed to load face database: {e}")
        face_db = {}
        exit()

    # Configure Performance Variables
    SIMILARITY_THRESHOLD = 0.45
    frame_skip_rate = 4
    frame_count = 0
    fps_smoothed = 0
    prev_time = time.time()
    
    # Saving Temporary face
    last_detected_faces = []
    is_under_attack = False
    attack_cooldown = 0

    print("[INFO] Running...")

    # Initialize UI
    app_instance = UI()

    cv2.namedWindow("Drone E99 Face Recognition and Anti Spoofing")

    while True:
        frame = camera.read()
        
        if frame is None:
            continue

        if frame_count % (frame_skip_rate + 1) == 0:
            faces = models.detect_and_recognize(frame)
            
            # Reset previous faces
            last_detected_faces = []

            is_under_attack = False
            
            # Take note of detected faces for drawing
            for face in faces:
                bbox = face.bbox.astype(int)
                feat = face.embedding / np.linalg.norm(face.embedding)

                max_sim = 0.0
                identity = "Unknown"
                for name, db_feat in face_db.items():
                    sim = np.dot(feat, db_feat)
                    if sim > max_sim:
                        max_sim = sim
                        identity = name

                # Compare with database
                if max_sim >= SIMILARITY_THRESHOLD:
                    display_name = f"{identity} ({max_sim:.2f})"
                    id_color = (0, 255, 0) 
                else:
                    display_name = f"Unknown ({max_sim:.2f})"
                    id_color = (0, 255, 255)

                # Liveness Check
                is_real, liveness_score = liveness.check_liveness(frame, bbox)
                if is_real:
                    liveness_label = "Real"
                    color = (0, 255, 0)
                else:
                    liveness_label = "Spoof"
                    color = (0, 0, 255)
                    is_under_attack = True

                # Check similarity threshold
                if max_sim >= SIMILARITY_THRESHOLD:
                    display_name = f"{identity} ({max_sim:.2f})"
                else:
                    display_name = f"Unknown ({max_sim:.2f})"

                last_detected_faces.append((bbox, display_name, liveness_label, liveness_score,id_color, color))

        # Make Bounding Box and lable
        for bbox, display_name, liveness_label, liveness_score, id_color, color in last_detected_faces:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), id_color, 2)
            cv2.putText(frame, display_name, (bbox[0], bbox[1] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 2)
            cv2.putText(frame, f"{liveness_label} ({liveness_score:.2f})", (bbox[0], bbox[1] -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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

        frame_count += 1

        frame = app_instance.process_ui(frame)

        # Show App
        cv2.imshow("Drone E99 Face Recognition and Anti Spoofing", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            app_instance.take_snapshot(frame)
        elif key == ord('r'):
            app_instance.toggle_recording()

    camera.stop()
    app_instance.cleanup()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()