import cv2
import os
from datetime import datetime
from camera_config import cameraDroneThread
from model import Models

def main():
    
    nama = input(">>> Enter Name: ").strip()
    if not nama:
        print("[ERROR] Name cannot be empty!")
        return

    # Create dataset folder
    save_dir = os.path.join("dataset", nama)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"[INFO] Folder created: {save_dir}")

    RTSP_URL = "rtsp://192.168.1.1:7070/webcam"
    print("[INFO] Connecting...")
    camera = cameraDroneThread(RTSP_URL).start()
    
    print("[INFO] Loading model...")
    models = Models()

    print(f" Input name: {nama}")
    print(" 1. Press 'S' for take picture.")
    print(" 2. Press 'Q' to exit.")

    frame_count = 0
    frame_skip = 4
    current_faces = []

    while True:
        frame = camera.read()
        if frame is None:
            continue

        # Save clean frame for snapshot
        clean_frame = frame.copy()

        # Detect and recognize faces
        if frame_count % (frame_skip + 1) == 0:
            current_faces = models.detect_and_recognize(frame)

        # Draw bounding boxes and labels
        for face in current_faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
            cv2.putText(frame, "Target", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(frame, f"Subjek: {nama}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'S' for PHOTO | 'Q' EXIT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Capture Dataset Drone", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if len(current_faces) == 0:
                print("[WARNING] Face not detected!")
            elif len(current_faces) > 1:
                print("[WARNING] More than one face detected!")
            else:
                # Save clean frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"drone_{timestamp}.jpg")
                cv2.imwrite(filename, clean_frame)
                
                # Hitung jumlah foto di folder
                jumlah_foto = len(os.listdir(save_dir))
                print(f"[SUCCESS] Picture {jumlah_foto} saved: {filename}")

        elif key == ord('q'):
            break

    camera.stop()
    cv2.destroyAllWindows()
    print("\n[INFO] Finish: Dataset saved to", save_dir)

if __name__ == "__main__":
    main()