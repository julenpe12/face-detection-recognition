import cv2
import numpy as np
import time
import collections
import psutil
import GPUtil
import torch

# Face detector using Haar cascade
class FaceDetectorHaar:
    def __init__(self, cascade_path="/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"):
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise ValueError("Cascade file not found or corrupt: {}".format(cascade_path))

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return faces

# Face recognizer using a TorchScript ArcFace model (no torchvision or PIL)
class FaceRecognizerArcFaceTorch:
    def __init__(self, path="models/arcface.pt", thresh=0.3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()
        self.features = []
        self.labels = []
        self.thresh = thresh

    def preprocess(self, roi):
        # Convert BGR to RGB
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # Resize to 112x112
        resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        # Convert to float32 in range [0,1]
        arr = resized.astype(np.float32) / 255.0
        # Normalize to [-1, +1]
        arr = (arr - 0.5) / 0.5
        # Rearrange to (C, H, W)
        arr = np.transpose(arr, (2, 0, 1))
        # Add batch dimension (1, C, H, W)
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

    def add(self, roi, label):
        embedding = self.extract(roi)
        self.features.append(embedding)
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

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        exit(1)

    # Set capture resolution to 640x480 for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

    print("[T] Train | [Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate system metrics
        now = time.time()
        if now > prev_time:
            fps = 1.0 / (now - prev_time)
        else:
            fps = 0.0
        prev_time = now
        buf_fps.append(fps)
        cpu = proc.cpu_percent() / psutil.cpu_count()
        buf_cpu.append(cpu)
        mem = proc.memory_info().rss / (1024 * 1024)
        buf_mem.append(mem)
        frame_count += 1
        if frame_count % 10 == 0:
            gpus = GPUtil.getGPUs()
            if gpus:
                load = gpus[0].load * 100.0
            else:
                load = 0.0
            buf_gpu.append(load)

        avg_fps = sum(buf_fps) / len(buf_fps) if buf_fps else 0.0
        avg_cpu = sum(buf_cpu) / len(buf_cpu) if buf_cpu else 0.0
        avg_mem = sum(buf_mem) / len(buf_mem) if buf_mem else 0.0
        avg_gpu = sum(buf_gpu) / len(buf_gpu) if buf_gpu else 0.0

        faces = det.detect(frame)

        if training_mode and len(faces) > 0:
            x, y, w, h = faces[0]
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue
            roi = frame[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            rec.add(roi, current_label)
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
                training_mode = False
                count = 0
                print("Training completed for '{}'.".format(current_label))

        else:
            if not rec.features:
                cv2.putText(
                    frame,
                    "No training data",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            for (x, y, w, h) in faces:
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

        # Overlay system metrics
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
            current_label = input("Label: ").strip()
            if current_label:
                training_mode = True
                count = 0
                print("Training mode: '{}'".format(current_label))

    cap.release()
    cv2.destroyAllWindows()
