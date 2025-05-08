import cv2
import numpy as np
import pickle
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

MODEL_FILENAME = "trained/mobilefacenet_model.pkl"  # File to save/load recognizer state
RECOGNITION_THRESHOLD = 0.3  # Cosine distance threshold

# =============================================================================
# Face Detector using YuNet
# =============================================================================
class FaceDetectorYuNet:
    """Face detector based on OpenCV's YuNet (ONNX)."""

    def __init__(self, model_path="models/face_detection_yunet_2023mar.onnx", score_threshold: float = 0.5):
        if not os.path.exists(model_path):
            raise ValueError(f"YuNet model not found: {model_path}")
        self.detector = cv2.FaceDetectorYN.create(model_path, "", (0, 0), score_threshold)

    def detect(self, image):
        """Detect faces in BGR image. Returns an ndarray of detections (may be empty)."""
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        rc, faces = self.detector.detect(image)
        if rc > 0 and faces is not None:
            return faces  # ndarray of shape (N, 15)
        return []

# =============================================================================
# Face Recognizer using MobileFaceNet TorchScript
# =============================================================================
class FaceRecognizerMobileFaceNetTorch:
    def __init__(self, model_path="models/mobilefacenet_scripted.pt", recognition_threshold=RECOGNITION_THRESHOLD, device="cpu"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.features_database = []  # List to store normalized embeddings.
        self.labels = []  # Corresponding labels for the embeddings.
        self.recognition_threshold = recognition_threshold

        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def preprocess(self, face_image):
        pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_image).unsqueeze(0)
        return tensor.to(self.device)

    def extract_features(self, face_image):
        tensor = self.preprocess(face_image)
        with torch.no_grad():
            embedding = self.model(tensor)
        embedding = embedding.flatten().cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def add_training_sample(self, face_image, label):
        embedding = self.extract_features(face_image)
        self.features_database.append(embedding)
        self.labels.append(label)

    def recognize(self, face_image):
        embedding = self.extract_features(face_image)
        if len(self.features_database) == 0:
            return "Unknown", 1.0
        similarities = np.array([np.dot(embedding, feat) for feat in self.features_database])
        distances = 1 - similarities
        min_index = int(np.argmin(distances))
        return self.labels[min_index], float(distances[min_index])

# =============================================================================
# Utility functions to save/load the recognizer state.
# =============================================================================
def save_model(recognizer, filename=MODEL_FILENAME):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(recognizer, f)
    print("Model state saved to", filename)

def load_model(filename=MODEL_FILENAME):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            recognizer = pickle.load(f)
        print("Model state loaded from", filename)
        return recognizer
    return None

# =============================================================================
# Main script: live camera face detection, training, and recognition
# =============================================================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    face_detector = FaceDetectorYuNet()
    face_recognizer = load_model() or FaceRecognizerMobileFaceNetTorch()

    training_mode = False
    target_training_samples = 250
    current_label = None
    training_sample_count = 0

    print("Press 'T' to start training mode with a new label. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_detector.detect(frame)

        if training_mode:
            if len(faces) > 0:
                x, y, w, h = faces[0][:4].astype(int)
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size:
                        face_recognizer.add_training_sample(face_roi, current_label)
                        training_sample_count += 1
                        cv2.putText(frame, f"Training {current_label}: {training_sample_count}/{target_training_samples}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if training_sample_count >= target_training_samples:
                training_mode = False
                training_sample_count = 0
                save_model(face_recognizer)
                print(f"Model updated with label '{current_label}'. Returning to recognition mode.")

        else:
            if not face_recognizer.features_database:
                cv2.putText(frame, "No training data", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue
                face_roi = frame[y:y+h, x:x+w]
                if not face_roi.size:
                    continue
                try:
                    label, dist = face_recognizer.recognize(face_roi)
                    txt_color = (0, 255, 0) if dist < face_recognizer.recognition_threshold else (0, 0, 255)
                    name = label if dist < face_recognizer.recognition_threshold else "Unknown"
                    cv2.putText(frame, f"{name} ({dist:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2)
                except Exception:
                    cv2.putText(frame, "Recognition error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("T"), ord("t")) and not training_mode:
            current_label = input("Enter label for training: ").strip()
            if current_label:
                training_mode = True
                training_sample_count = 0
                print(f"Training mode activated for label '{current_label}'. Collecting training images...")
            else:
                print("Empty label. Training mode not activated.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()