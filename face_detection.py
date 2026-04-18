import cv2
from insightface.app import FaceAnalysis

class Models:
    def __init__(self):
        print("[INFO] Loading model...")
        # Activate Face Detection model
        self.face_app = FaceAnalysis(name='buffalo_s', allowed_modules=['detection'])
        
        # Configure resolution
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))

    def detect_faces(self, frame_bgr):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_app.get(rgb_frame)
        return faces