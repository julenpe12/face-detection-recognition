import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATH = "models/yolov8n-face.pt"
DEVICE_ID = 0
RECOGNITION_THRESHOLD = 0.3
TARGET_SAMPLES = 250

# Reconocedor ArcFace actualizado con detección YOLO
class FaceRecognizerArcFaceTorch:
    def __init__(self, model_path="models/arcface.pt", recognition_threshold=RECOGNITION_THRESHOLD, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.features_database = []
        self.labels = []
        self.recognition_threshold = recognition_threshold

    def preprocess(self, face_image):
        # Input: face_image is a BGR numpy array (H×W×3 uint8)
        # 1) Convert BGR → RGB
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        # 2) Resize to (112,112)
        resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        # 3) Convert to float32 in [0,1]
        arr = resized.astype(np.float32) / 255.0
        # 4) Normalize to [-1, +1]
        arr = (arr - 0.5) / 0.5
        # 5) Transpose H×W×C → C×H×W
        arr = np.transpose(arr, (2, 0, 1))
        # 6) Convert to torch tensor, añadir batch
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        return tensor


    def extract_features(self, face_image):
        tensor = self.preprocess(face_image)
        with torch.no_grad():
            emb = self.model(tensor).flatten().cpu().numpy()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def add_training_sample(self, face_image, label):
        emb = self.extract_features(face_image)
        self.features_database.append(emb)
        self.labels.append(label)

    def recognize(self, face_image):
        emb = self.extract_features(face_image)
        if not self.features_database:
            return "Unknown", 1.0
        sims = np.dot(self.features_database, emb)
        dists = 1 - sims
        idx = np.argmin(dists)
        return self.labels[idx], dists[idx]

if __name__ == "__main__":
    # Carga del modelo YOLO para detección de caras
    yolo = YOLO(MODEL_PATH).to('cuda' if torch.cuda.is_available() else 'cpu')
    cap = cv2.VideoCapture(DEVICE_ID)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    recognizer = FaceRecognizerArcFaceTorch()
    training_mode = False
    sample_count = 0
    current_label = None

    print("[T] Entrenar nuevo label | [q] Salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detección de caras con YOLO
        results = yolo(frame)
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1, y1, x2 - x1, y2 - y1))

        if training_mode and boxes:
            x, y, w, h = boxes[0]
            roi = frame[y:y+h, x:x+w]
            recognizer.add_training_sample(roi, current_label)
            sample_count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Train {current_label}: {sample_count}/{TARGET_SAMPLES}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if sample_count >= TARGET_SAMPLES:
                training_mode = False
                sample_count = 0
                print(f"Entrenamiento completado para '{current_label}'")
        else:
            for x, y, w, h in boxes:
                roi = frame[y:y+h, x:x+w]
                label, dist = recognizer.recognize(roi)
                color = (0, 255, 0) if dist < recognizer.recognition_threshold else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{label} ({dist:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Feed YOLO", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('t'):
            lbl = input("Label: ").strip()
            if lbl:
                current_label = lbl
                training_mode = True
                print(f"Modo entrenamiento: '{current_label}'")

    cap.release()
    cv2.destroyAllWindows()