import cv2
import numpy as np
import onnxruntime as ort

class MiniFASNetV2:
    def __init__ (self, model_path="models/MiniFASNetV2.onnx"):
        print("[INFO]: Loading MiniFASNetV2 model...")
        try:
            self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
            print("[INFO]: MiniFASNetV2 model loaded successfully.")

        except Exception as e:
            print(f"[ERROR] Failed to load MiniFASNetV2 model: {e}")
            exit()

    def check_liveness(self, frame_bgr, bbox):
        src_h, src_w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        box_w, box_h = x2 - x1, y2 - y1
        
        scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, 2.7)
        new_w, new_h = box_w * scale, box_h * scale
        center_x, center_y = x1 + box_w / 2, y1 + box_h / 2

        nx1, ny1 = max(0, int(center_x - new_w / 2)), max(0, int(center_y - new_h / 2))
        nx2, ny2 = min(src_w - 1, int(center_x + new_w / 2)), min(src_h - 1, int(center_y + new_h / 2))

        face_crop = frame_bgr[ny1:ny2 + 1, nx1:nx2 + 1]
        if face_crop.size == 0: return False, 0.0

        # Preprocessing for MiniFASNetV2
        face_crop = cv2.resize(face_crop, (80, 80))
        face_tensor = face_crop.astype(np.float32)
        face_tensor = np.transpose(face_tensor, (2, 0, 1))
        face_tensor = np.expand_dims(face_tensor, axis=0)

        # Inference
        input_name = self.session.get_inputs()[0].name
        logits = self.session.run(None, {input_name: face_tensor})[0][0]
        
        e_x = np.exp(logits - np.max(logits))
        probs = e_x / e_x.sum()

        real_score = float(probs[1]) 
        is_real = real_score > 0.85 

        return is_real, real_score