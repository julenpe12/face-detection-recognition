import cv2
import numpy as np
import time
import collections
import psutil
import GPUtil

# Ruta al archivo Haar Cascade (debe descargarse y colocarse en el mismo directorio)
CASCADE_PATH = "models/haarcascade_frontalface_default.xml"

MODEL_PATH = "models/arcface_int8.onnx"
RECOGNITION_THRESHOLD = 0.3

class FaceDetectorHaar:
    def __init__(self, cascade_path=CASCADE_PATH):
        if not cv2.os.path.exists(cascade_path):
            raise ValueError(f"Cascade file not found: {cascade_path}")
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray)
        return faces

class FaceRecognizerArcFace:
    def __init__(self, model_path=MODEL_PATH, threshold=RECOGNITION_THRESHOLD):
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.features_database = []
        self.labels = []
        self.threshold = threshold

    def preprocess(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (112, 112))
        return cv2.dnn.blobFromImage(resized)

    def extract(self, img):
        blob = self.preprocess(img)
        self.model.setInput(blob)
        emb = self.model.forward().flatten()
        norm = np.linalg.norm(emb)
        return (emb / norm) if norm > 0 else emb

    def add(self, img, label):
        emb = self.extract(img)
        self.features_database.append(emb)
        self.labels.append(label)

    def recognize(self, img):
        emb = self.extract(img)
        if not self.features_database:
            return "Unknown", 1.0
        sims = np.dot(np.array(self.features_database), emb)
        dists = 1 - sims
        idx = np.argmin(dists)
        return self.labels[idx], dists[idx]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara.")
        return

    detector = FaceDetectorHaar()
    recognizer = FaceRecognizerArcFace()

    process = psutil.Process(os.getpid())
    logical_cores = psutil.cpu_count(logical=True)

    fps_buffer = collections.deque(maxlen=30)
    cpu_buffer = collections.deque(maxlen=30)
    mem_buffer = collections.deque(maxlen=30)
    gpu_buffer = collections.deque(maxlen=30)
    inf_buffer = collections.deque(maxlen=30)

    prev_time = time.time()
    training_mode = False
    current_label = None
    sample_count = 0
    target_samples = 250

    print("Presione 'T' para iniciar entrenamiento, 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Cálculo de métricas
        now = time.time()
        fps_buffer.append(1.0 / (now - prev_time))
        prev_time = now

        cpu_usage = process.cpu_percent(interval=None) / logical_cores
        cpu_buffer.append(cpu_usage)

        mem_usage = process.memory_info().rss / (1024 * 1024)
        mem_buffer.append(mem_usage)

        if len(fps_buffer) % 10 == 0:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_load = gpus[0].load * 100.0
                gpu_buffer.append(gpu_load)
            else:
                gpu_buffer.append(0.0)

        avg_fps = sum(fps_buffer) / len(fps_buffer)
        avg_cpu = sum(cpu_buffer) / len(cpu_buffer)
        avg_mem = sum(mem_buffer) / len(mem_buffer)
        avg_gpu = sum(gpu_buffer) / len(gpu_buffer) if gpu_buffer else 0.0

        faces = detector.detect(frame)

        if training_mode and len(faces) > 0:
            x, y, w, h = faces[0]
            if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                continue
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            start_inf = time.time()
            recognizer.add(roi, current_label)
            inf_buffer.append((time.time() - start_inf) * 1000)
            sample_count += 1
            if sample_count >= target_samples:
                training_mode = False
                sample_count = 0
                print(f"Entrenamiento completado para '{current_label}'.")
        else:
            for (x, y, w, h) in faces:
                if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                    continue
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                start_inf = time.time()
                lbl, dist = recognizer.recognize(roi)
                inf_buffer.append((time.time() - start_inf) * 1000)
                color = (0, 255, 0) if dist < recognizer.threshold else (0, 0, 255)
                cv2.putText(
                    frame,
                    f"{lbl} ({dist:.2f})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Superponer métricas (esquina superior izquierda)
        avg_inf = sum(inf_buffer) / len(inf_buffer) if inf_buffer else 0.0
        cv2.putText(
            frame,
            f"FPS: {avg_fps:.0f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Inf: {avg_inf:.0f} ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"CPU: {avg_cpu*100:.1f}% per core",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"RAM: {avg_mem:.2f} MB",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"GPU: {avg_gpu:.1f}%",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 255, 0),
            2
        )

        cv2.imshow("Feed Metrics", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in (ord('T'), ord('t')) and not training_mode:
            current_label = input("Etiqueta: ").strip()
            if current_label:
                training_mode = True
                sample_count = 0
                print(f"Modo entrenamiento: '{current_label}'")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
