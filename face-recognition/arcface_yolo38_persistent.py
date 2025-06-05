import cv2
import numpy as np
import torch
import pickle
import os
from ultralytics import YOLO
from pathlib import Path

# Configuración de captura (anchura × altura)
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

MODEL_PATH = "models/yolov8n-face.pt"
DEVICE_ID = 0
MODEL_FILE = "trained/arcface_db.pkl"
TARGET_SAMPLES = 250

class FaceRecognizerArcFaceTorch:
    def __init__(self, path="models/arcface.pt", thresh=0.3):
        if not os.path.exists(path):
            raise ValueError("TorchScript model not found: {}".format(path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()
        self.features = []
        self.labels = []
        self.thresh = thresh
        self.model_path = path

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        if "model_path" not in state:
            state["model_path"] = "models/arcface.pt"
        self.__dict__.update(state)
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()

    def extract(self, roi):
        # 1) Convert BGR → RGB
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # 2) Resize a 112×112
        resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        # 3) Convertir a float32 en [0,1]
        arr = resized.astype(np.float32) / 255.0
        # 4) Normalizar a [-1,+1]
        arr = (arr - 0.5) / 0.5
        # 5) Transponer H×W×C → C×H×W
        arr = np.transpose(arr, (2, 0, 1))
        # 6) Añadir dimensión de batch y mover a dispositivo
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        # 7) Forward pass
        with torch.no_grad():
            v = self.model(tensor).flatten().cpu().numpy()
        # 8) Normalizar embedding
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def add(self, roi, label):
        self.features.append(self.extract(roi))
        self.labels.append(label)

    def recognize(self, roi):
        if not self.features:
            return "Unknown", 1.0
        emb = self.extract(roi)
        db = np.vstack(self.features)
        sims = np.dot(db, emb)
        dists = 1.0 - sims
        idx = np.argmin(dists)
        return self.labels[idx], float(dists[idx])

def save_model(recognizer, path=MODEL_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(recognizer, f)
    print("Database saved to '{}'".format(path))

def load_model(path=MODEL_FILE):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

if __name__ == "__main__":
    yolo = YOLO(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture(DEVICE_ID)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    # Ajustar resolución de captura
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    rec = load_model() or FaceRecognizerArcFaceTorch(path="models/arcface.pt", thresh=0.3)
    mode = False
    cnt = 0
    lbl = None

    print("[T] Train  |  [Q] Quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # — Detección de YOLO —
        results = yolo(frame)
        boxes = []
        for box in results[0].boxes:
            x1 = int(box.xyxy[0][0])
            y1 = int(box.xyxy[0][1])
            x2 = int(box.xyxy[0][2])
            y2 = int(box.xyxy[0][3])
            w = x2 - x1
            h = y2 - y1
            boxes.append((x1, y1, w, h))

        if mode and boxes:
            x, y, w, h = boxes[0]
            roi = frame[y : y + h, x : x + w]
            rec.add(roi, lbl)
            cnt += 1
            if cnt >= TARGET_SAMPLES:
                mode = False
                cnt = 0
                save_model(rec)
                print("Saved label '{}'".format(lbl))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame,
                "{}/{}".format(cnt, TARGET_SAMPLES),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
        else:
            for (x, y, w, h) in boxes:
                roi = frame[y : y + h, x : x + w]
                label, dist = rec.recognize(roi)
                color = (0, 255, 0) if dist < rec.thresh else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    "{} ({:.2f})".format(label, dist),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

        cv2.imshow("Feed YOLO Persistent", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t"):
            lbl = input("Label: ").strip()
            if lbl:
                mode = True

    cap.release()
    cv2.destroyAllWindows()
