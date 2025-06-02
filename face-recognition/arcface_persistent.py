import cv2
import numpy as np
import pickle
import os

# Path to Haar Cascade (download and place alongside this script)
CASCADE_PATH = "models/haarcascade_frontalface_default.xml"

MODEL_PATH = "models/arcface_int8.onnx"
DB_FILE    = "trained/arcface_db.pkl"
RECOGNITION_THRESHOLD = 0.3

class FaceDetectorHaar:
    def __init__(self, cascade_path=CASCADE_PATH, scaleFactor=1.1, minNeighbors=5):
        if not os.path.exists(cascade_path):
            raise ValueError("Cascade file not found or corrupt: {}".format(cascade_path))
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors
        )

class FaceRecognizerArcFace:
    def __init__(self, model_path=MODEL_PATH, threshold=RECOGNITION_THRESHOLD):
        if not os.path.exists(model_path):
            raise ValueError("ONNX model not found: {}".format(model_path))
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.features_database = []
        self.labels = []
        self.threshold = threshold

    def preprocess(self, face_image):
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        return cv2.dnn.blobFromImage(resized)

    def extract(self, face_image):
        blob = self.preprocess(face_image)
        self.model.setInput(blob)
        emb = self.model.forward().flatten()
        norm = np.linalg.norm(emb)
        return (emb / norm) if norm > 0 else emb

    def add_sample(self, face_image, label):
        emb = self.extract(face_image)
        self.features_database.append(emb)
        self.labels.append(label)

    def recognize(self, face_image):
        emb = self.extract(face_image)
        if not self.features_database:
            return "Unknown", 1.0
        sims = np.dot(np.array(self.features_database), emb)
        dists = 1 - sims
        idx = np.argmin(dists)
        return self.labels[idx], dists[idx]

def save_db(recognizer, path=DB_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "features": recognizer.features_database,
            "labels": recognizer.labels
        }, f)
    print("Database saved.")

def load_db(recognizer, path=DB_FILE):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        recognizer.features_database = data["features"]
        recognizer.labels = data["labels"]
        print("Database loaded.")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    # Set capture resolution to 640×480 for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = FaceDetectorHaar()
    recognizer = FaceRecognizerArcFace()
    load_db(recognizer)

    training_mode = False
    current_label = None
    sample_count = 0
    target_samples = 250

    print("Press 'T' to start training mode, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        if training_mode and len(faces) > 0:
            x, y, w, h = faces[0]
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            recognizer.add_sample(roi, current_label)
            sample_count += 1
            if sample_count >= target_samples:
                save_db(recognizer)
                training_mode = False
                sample_count = 0
                print("Training completed for '{}'.".format(current_label))
        else:
            for (x, y, w, h) in faces:
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                lbl, dist = recognizer.recognize(roi)
                color = (0, 255, 0) if dist < recognizer.threshold else (0, 0, 255)
                cv2.putText(
                    frame,
                    "{} ({:.2f})".format(lbl, dist),
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.imshow("Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in (ord('T'), ord('t')) and not training_mode:
            current_label = input("Enter label: ").strip()
            if current_label:
                training_mode = True
                sample_count = 0
                print("Training mode activated for '{}'.".format(current_label))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
