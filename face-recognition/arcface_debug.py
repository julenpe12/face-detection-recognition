import cv2
import numpy as np
import time
import collections
import psutil
import GPUtil
import torch

# -------------------------------------------------------------------
# Detector de caras con Haar Cascade
# -------------------------------------------------------------------
class FaceDetectorHaar:
    def __init__(self, cascade_path="/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"):
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise ValueError("Cascade file not found o corrupto: {}".format(cascade_path))

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return faces

# -------------------------------------------------------------------
# Reconocedor ArcFace con PyTorch (sin torchvision ni PIL)
# -------------------------------------------------------------------
class FaceRecognizerArcFaceTorch:
    def __init__(self, path="models/arcface.pt", thresh=0.3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Carga modelo TorchScript (por ejemplo, torch.jit.save)
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()
        self.features = []
        self.labels = []
        self.thresh = thresh

    def preprocess(self, roi):
        """
        Preprocesado sin torchvision:
         1) Asegurar que roi es un arreglo uint8 BGR.
         2) Convertir BGR → RGB.
         3) Redimensionar a 112×112.
         4) Convertir a float32 en rango [-1, +1]: ((img/255) - 0.5) / 0.5
         5) Reordenar canales a (C, H, W) y añadir batch.
        """
        # 1) Convertir BGR → RGB
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # 2) Redimensionar a 112×112
        resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        # 3) Convertir a float32 y normalizar a [-1, +1]
        #    Primero pasamos a [0,1]:
        arr = resized.astype(np.float32) / 255.0
        #    Luego (x - 0.5) / 0.5 → [-1, +1]
        arr = (arr - 0.5) / 0.5
        # 4) Reordenar a (C, H, W)
        #    arr tiene forma (112,112,3), canales al final
        arr = np.transpose(arr, (2, 0, 1))
        # 5) Añadir dimensión de batch → (1, C, H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        return tensor

    def extract(self, roi):
        """
        Dado un ROI en formato BGR uint8, retorna el embedding normalizado (1D numpy array).
        """
        inp = self.preprocess(roi)  # → Tensor (1, 3, 112, 112)
        with torch.no_grad():
            emb = self.model(inp)               # Supone que la red devuelve (1, 512) por ejemplo
            emb = emb.view(-1).cpu().numpy()    # Vector 1D en CPU
        norm = np.linalg.norm(emb)
        if norm > 0:
            return emb / norm
        else:
            return emb

    def add(self, roi, label):
        embedding = self.extract(roi)
        self.features.append(embedding)
        self.labels.append(label)

    def recognize(self, roi):
        if not self.features:
            return "Unknown", 1.0
        emb = self.extract(roi)
        db = np.vstack(self.features)             # (N, embedding_size)
        sims = np.dot(db, emb)                    # (N,)
        dists = 1.0 - sims                         # distancia coseno
        idx = np.argmin(dists)
        return self.labels[idx], float(dists[idx])

# -------------------------------------------------------------------
# Bucle principal: captura de cámara, métricas y reconocimiento
# -------------------------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara.")
        exit(1)

    proc = psutil.Process()
    buf_fps = collections.deque(maxlen=30)
    buf_cpu = collections.deque(maxlen=30)
    buf_mem = collections.deque(maxlen=30)
    buf_gpu = collections.deque(maxlen=30)
    prev_time = time.time()
    frame_count = 0

    det = FaceDetectorHaar()
    rec = FaceRecognizerArcFaceTorch(path="models/arcface.pt", thresh=0.3)

    training_mode = False
    target_samples = 250
    current_label = None
    count = 0

    print("[T] Entrenar | [Q] Salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # — Métricas de sistema —
        now = time.time()
        if now > prev_time:
            fps = 1.0 / (now - prev_time)
        else:
            fps = 0.0
        prev_time = now
        buf_fps.append(fps)
        cpu = proc.cpu_percent() / psutil.cpu_count()
        buf_cpu.append(cpu)
        mem = proc.memory_info().rss / (1024 * 1024)  # MB
        buf_mem.append(mem)
        frame_count += 1
        if frame_count % 10 == 0:
            gpus = GPUtil.getGPUs()
            if gpus:
                load = gpus[0].load * 100.0
            else:
                load = 0.0
            buf_gpu.append(load)

        if buf_fps:
            avg_fps = sum(buf_fps) / len(buf_fps)
        else:
            avg_fps = 0.0
        if buf_cpu:
            avg_cpu = sum(buf_cpu) / len(buf_cpu)
        else:
            avg_cpu = 0.0
        if buf_mem:
            avg_mem = sum(buf_mem) / len(buf_mem)
        else:
            avg_mem = 0.0
        if buf_gpu:
            avg_gpu = sum(buf_gpu) / len(buf_gpu)
        else:
            avg_gpu = 0.0

        # — Detección de caras —
        faces = det.detect(frame)

        if training_mode and len(faces) > 0:
            x, y, w, h = faces[0]
            # Validar límites
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue
            roi = frame[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            rec.add(roi, current_label)
            count += 1
            cv2.putText(
                frame,
                "Entrenando '{}': {}/{}".format(current_label, count, target_samples),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if count >= target_samples:
                training_mode = False
                count = 0
                print("Entrenamiento completado para '{}'.".format(current_label))

        else:
            if not rec.features:
                cv2.putText(
                    frame,
                    "No hay datos de entrenamiento",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            for (x, y, w, h) in faces:
                # Validar límites
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue
                roi = frame[y : y + h, x : x + w]
                if roi.size == 0:
                    continue
                label, dist = rec.recognize(roi)
                if dist < rec.thresh:
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

        cv2.imshow("Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t"):
            current_label = input("Etiqueta: ").strip()
            if current_label:
                training_mode = True
                count = 0
                print("Modo entrenamiento: '{}'".format(current_label))

    cap.release()
    cv2.destroyAllWindows()
