import cv2
import os
import pickle
import numpy as np
from insightface.app import FaceAnalysis

def main():
    print("[INFO] Load model...")
    app = FaceAnalysis(name='buffalo_s')
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    dataset_dir = "dataset"
    face_db = {}
    
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Folder '{dataset_dir}' not found.")
        return

    print("[INFO] Dataset Processing...")
    
    # Looping each person folder in dataset
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        embeddings = []
        # Looping each image in person's folder
        for image_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"[WARNING] Cannot read image: {img_path}")
                continue
                
            # Detect and extract face
            faces = app.get(img)
            
            if len(faces) == 0:
                print(f"[WARNING] No face detected in: {img_path}")
                continue
                
            embedding = faces[0].embedding
            embeddings.append(embedding)
            print(f"  -> SUCCESS to extract embedding from {image_name}")
            
        # Few shot learning
        if len(embeddings) > 0:
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Normalization L2
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            # Save to database
            face_db[person_name] = avg_embedding
            print(f"[SUCCESS] Successfully registered '{person_name}' with {len(embeddings)} reference photos.\n")

    # Save database to file
    with open("face_db.pkl", "wb") as f:
        pickle.dump(face_db, f)
        
    print("[INFO] Process completed. Face database successfully saved as 'face_db.pkl'.")

if __name__ == "__main__":
    main()