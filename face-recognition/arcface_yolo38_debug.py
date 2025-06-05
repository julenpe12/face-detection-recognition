import cv2
import numpy as np
import time
import collections
import psutil
import GPUtil
import torch
from ultralytics import YOLO

# Configuración de captura (anchura × altura)
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

MODEL_PATH = "models/yolov8n-face.pt"
DEVICE_ID = 0
TARGET_SAMPLES = 250

class FaceRecognizerArcFaceTorch:
    def __init__(self, path="models/arcface.pt", thresh=0.3):
        if not torch.cuda.is_available() and not os.path.exists(path):
            raise ValueError("TorchScript model not found: {}".format(path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()
        self.features = []
        self.labels = []
        self.thresh = thresh

    def extract(self, roi):
        # 1) Convert BGR → RGB
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # 2) Redimensionar a 112×112
        resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        # 3) Convertir a float32 en [0,1]
        arr = resized.astype(np.float32) / 255.0
        # 4) Normalizar a [-1,+1]
        arr = (arr - 0.5) / 0.5
        # 5) Transponer H×W×C → C×H×W
        arr = np.transpose(arr, (2, 0, 1))
        # 6) Añadir dimensión de batch y mover a dispositivo
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        # 7) Obtener embedding
        with torch.no_grad():
            v = self.model(tensor).flatten().cpu().numpy()
        # 8) Normalizar a longitud 1
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def add(self, roi, label):
        self.features.append(self.extract(roi))
        self.labels.append(label)

    def recognize(self, roi):
        if not self.features:
            return "Unknown", 1.0
        emb = self.extract(roi)
        db = np.vstack(self.features)           # (N, 512)
        sims = np.dot(db, emb)                  # similitud coseno
        dists = 1.0 - sims                       # distancia
        idx = np.argmin(dists)
        return self.labels[idx], float(dists[idx])

if __name__ == "__main__":
    yolo = YOLO(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture(DEVICE_ID)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    # Ajustar resolución de captura
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    proc = psutil.Process()
    buf_fps = collections.deque(maxlen=30)
    buf_cpu = collections.deque(maxlen=30)
    buf_mem = collections.deque(maxlen=30)
    buf_gpu = collections.deque(maxlen=30)
    prev_time = time.time()
    frame_count = 0

    rec = FaceRecognizerArcFaceTorch(path="models/arcface.pt", thresh=0.3)
    mode = False
    cnt = 0
    lbl = None

    print("[T] Train  |  [Q] Quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # — Cálculo de métricas de sistema —
        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now
        buf_fps.append(fps)
        buf_cpu.append(proc.cpu_percent() / psutil.cpu_count())
        buf_mem.append(proc.memory_info().rss / (1024 * 1024))  # en MB
        frame_count += 1
        if frame_count % 10 == 0:
            gpus = GPUtil.getGPUs()
            load = gpus[0].load * 100.0 if gpus else 0.0
            buf_gpu.append(load)

        avg_fps = sum(buf_fps) / len(buf_fps) if buf_fps else 0.0
        avg_cpu = sum(buf_cpu) / len(buf_cpu) if buf_cpu else 0.0
        avg_mem = sum(buf_mem) / len(buf_mem) if buf_mem else 0.0
        avg_gpu = sum(buf_gpu) / len(buf_gpu) if buf_gpu else 0.0

        # — Detección de caras con YOLO —
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

        # — Modo entrenamiento o reconocimiento —
        if mode and boxes:
            x, y, w, h = boxes[0]
            roi = frame[y : y + h, x : x + w]
            rec.add(roi, lbl)
            cnt += 1
            if cnt >= TARGET_SAMPLES:
                mode = False
                cnt = 0
                print("Entrenamiento completado para '{}'".format(lbl))
            # Dibujar durante entrenamiento
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

        # — Overlay de métricas —
        cv2.putText(
            frame,
            "FPS: {}".format(int(avg_fps)),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "CPU: {:.1f}%".format(avg_cpu),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Mem: {:.1f}MB".format(avg_mem),
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "GPU: {:.1f}%".format(avg_gpu),
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 255, 0),
            2,
        )

        cv2.imshow("Feed YOLO Metrics", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t"):
            lbl = input("Label: ").strip()
            if lbl:
                mode = True

    cap.release()
    cv2.destroyAllWindows()
