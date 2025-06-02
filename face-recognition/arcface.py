import os
import cv2
import numpy as np
import onnxruntime as ort

# Path to Haar Cascade (download and place alongside this script)
CASCADE_PATH = "models/haarcascade_frontalface_default.xml"
MODEL_PATH   = "models/arcfaceresnet100-8.onnx"   # Path to ArcFace ONNX model
THRESHOLD    = 0.3                                # Cosine distance threshold

class FaceDetectorHaar:
    def __init__(self, cascade_path=CASCADE_PATH, scaleFactor=1.1, minNeighbors=5):
        if not os.path.exists(cascade_path):
            raise ValueError("Cascade file not found or corrupt: {}".format(cascade_path))
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors
        )
        return faces

class FaceRecognizerArcFace:
    def __init__(self, model_path=MODEL_PATH, threshold=THRESHOLD):
        if not os.path.exists(model_path):
            raise ValueError("ONNX model not found: {}".format(model_path))
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.features_database = []   # List of normalized embeddings
        self.labels = []              # Corresponding labels
        self.threshold = threshold

    def preprocess(self, face_img):
        # Convert BGR→RGB, resize to 112×112, normalize to [0,1], reorder to (1,3,112,112)
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        return arr

    def extract_features(self, face_img):
        blob = self.preprocess(face_img)               # → (1,3,112,112) float32
        blob = np.ascontiguousarray(blob)              # Ensure contiguity

        print("DEBUG extract_features → blob.shape: {}, blob.dtype: {}".format(blob.shape, blob.dtype))
        try:
            outputs = self.session.run(None, {self.input_name: blob})
        except Exception as e:
            print("ERROR in session.run with real ROI:", e)
            dummy = np.zeros_like(blob, dtype=np.float32)
            try:
                _ = self.session.run(None, {self.input_name: dummy})
                print("DEBUG: dummy works, problem is with real ROI data.")
            except Exception as e2:
                print("ERROR dummy after ROI failure:", e2)
            return np.zeros(512, dtype=np.float32)

        embedding = outputs[0].reshape(-1)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def add_training_sample(self, face_img, label):
        emb = self.extract_features(face_img)
        self.features_database.append(emb)
        self.labels.append(label)

    def recognize(self, face_img):
        emb = self.extract_features(face_img)
        if not self.features_database:
            return "Unknown", 1.0
        db = np.vstack(self.features_database)         # shape: (N, embedding_size)
        sims = np.dot(db, emb)                         # shape: (N,)
        dists = 1.0 - sims                             # cosine distance
        idx = np.argmin(dists)
        return self.labels[idx], float(dists[idx])

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

    training_mode = False
    target_samples = 250
    current_label = None
    count = 0

    print("Press 'T' to start training mode with a new label. Press 'q' to quit.")

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
            recognizer.add_training_sample(roi, current_label)
            count += 1
            cv2.putText(
                frame,
                "Training '{}': {}/{}".format(current_label, count, target_samples),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
            )
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if count >= target_samples:
                training_mode = False
                count = 0
                print("Training completed for '{}'.".format(current_label))

        else:
            if not recognizer.features_database:
                cv2.putText(
                    frame,
                    "No training data",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
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
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.imshow("Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in (ord('T'), ord('t')) and not training_mode:
            current_label = input("Enter label for training: ").strip()
            if current_label:
                training_mode = True
                count = 0
                print("Training mode activated for label '{}'.".format(current_label))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
