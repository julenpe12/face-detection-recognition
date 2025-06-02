import cv2
import numpy as np
import torch
import pickle
import os

# Configuration: adjust capture resolution to match model input scaling requirements
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

# Face detector using Haar Cascade
class FaceDetectorHaar:
    def __init__(self, cascade_path="/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"):
        if not os.path.exists(cascade_path):
            raise ValueError("Cascade file not found or corrupt: {}".format(cascade_path))
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return faces

# Face recognizer using a TorchScript ArcFace model (no torchvision, no PIL)
class FaceRecognizerArcFaceTorch:
    def __init__(self, model_path="models/arcface.pt", threshold=0.3):
        if not os.path.exists(model_path):
            raise ValueError("TorchScript model not found: {}".format(model_path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.features = []
        self.labels = []
        self.threshold = threshold

    def preprocess(self, roi):
        # Convert BGR to RGB
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # Resize to 112×112
        resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        # Convert to float32 in [0,1]
        arr = resized.astype(np.float32) / 255.0
        # Normalize to [-1,+1]
        arr = (arr - 0.5) / 0.5
        # Rearrange to (C, H, W)
        arr = np.transpose(arr, (2, 0, 1))
        # Add batch dimension → (1, C, H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        return tensor

    def extract(self, roi):
        inp = self.preprocess(roi)
        with torch.no_grad():
            emb = self.model(inp)
            emb = emb.view(-1).cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            return emb / norm
        return emb

    def add_sample(self, roi, label):
        embedding = self.extract(roi)
        self.features.append(embedding)
        self.labels.append(label)

    def recognize(self, roi):
        if not self.features:
            return "Unknown", 1.0
        emb = self.extract(roi)
        db = np.vstack(self.features)            # shape: (N, embedding_size)
        sims = np.dot(db, emb)                   # shape: (N,)
        dists = 1.0 - sims                        # cosine distance
        idx = np.argmin(dists)
        return self.labels[idx], float(dists[idx])

def save_database(recognizer, path="trained/arcface_db.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "features": recognizer.features,
        "labels": recognizer.labels
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print("Database saved.")

def load_database(recognizer, path="trained/arcface_db.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        recognizer.features = data.get("features", [])
        recognizer.labels = data.get("labels", [])
        print("Database loaded.")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    # Set capture resolution from configuration
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    detector = FaceDetectorHaar()
    recognizer = FaceRecognizerArcFaceTorch(model_path="models/arcface.pt", threshold=0.3)
    load_database(recognizer)

    training_mode = False
    target_samples = 250
    current_label = None
    count = 0

    print("[T] Train  |  [Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        if training_mode and len(faces) > 0:
            x, y, w, h = faces[0]
            # Validate bounds
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue
            roi = frame[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            recognizer.add_sample(roi, current_label)
            count += 1
            cv2.putText(
                frame,
                "Training '{}': {}/{}".format(current_label, count, target_samples),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if count >= target_samples:
                save_database(recognizer)
                training_mode = False
                count = 0
                print("Training completed for '{}'.".format(current_label))
        else:
            for (x, y, w, h) in faces:
                # Validate bounds
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue
                roi = frame[y : y + h, x : x + w]
                if roi.size == 0:
                    continue
                label, dist = recognizer.recognize(roi)
                if dist < recognizer.threshold:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                cv2.putText(
                    frame,
                    "{} ({:.2f})".format(label, dist),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t") and not training_mode:
            current_label = input("Label: ").strip()
            if current_label:
                training_mode = True
                count = 0
                print("Entered training mode for '{}'.".format(current_label))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
