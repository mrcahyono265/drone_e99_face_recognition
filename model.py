import cv2
import numpy as np
from insightface.app import FaceAnalysis

class Models:
    def __init__(self):
        print("[INFO] Loading model...")
        # Activate Face Detection model
        self.face_app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection', 'recognition'])
        
        # Configure resolution
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))

    def detect_and_recognize(self, frame_bgr):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_app.get(rgb_frame)
        return faces
    
    def compute_sim(self, feat1, feat2):
        # Using cosine similarity
        return np.dot(feat1, feat2)