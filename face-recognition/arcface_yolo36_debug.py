import cv2
import numpy as np
import time
import collections
import psutil
import GPUtil
import torch
import os
import onnxruntime as ort
from pathlib import Path

# Configuration: capture resolution (w × h)
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

# YOLO configuration (unchanged)
MODEL_PATH = "models/yolov8n-face.onnx"
INPUT_SIZE = 640
CONF_THRES = 0.25
NMS_THRES = 0.45
DEVICE_ID = 0

def letterbox(img, new_size=INPUT_SIZE, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top : top + nh, left : left + nw] = img_resized
    return canvas, scale, left, top

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def iou(box, boxes):
    inter = (
        np.maximum(0, np.minimum(boxes[:, 2], box[2]) - np.maximum(boxes[:, 0], box[0]))
        * np.maximum(0, np.minimum(boxes[:, 3], box[3]) - np.maximum(boxes[:, 1], box[1]))
    )
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter + 1e-6
    return inter / union

def non_max_suppression(boxes, scores, iou_thres=NMS_THRES):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep

def preprocess_yolo(frame):
    img, scale, left, top = letterbox(frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return img, scale, left, top

def postprocess_yolo(pred, scale, left, top, orig_shape):
    pred = np.squeeze(pred).transpose(1, 0)
    boxes = xywh2xyxy(pred[:, :4])
    scores = pred[:, 4]
    mask = scores > CONF_THRES
    boxes, scores = boxes[mask], scores[mask]
    if boxes.size == 0:
        return []
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes /= scale
    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    keep = non_max_suppression(boxes, scores)
    return [(boxes[i].astype(int), float(scores[i])) for i in keep]

def get_execution_providers():
    pref = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    avail = ort.get_available_providers()
    return [p for p in pref if p in avail]

class FaceDetectorYOLO:
    def __init__(self, model_path=MODEL_PATH):
        if not Path(model_path).is_file():
            raise FileNotFoundError("ONNX model not found: {}".format(model_path))
        self.sess = ort.InferenceSession(model_path, providers=get_execution_providers())
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def detect(self, frame):
        img, scale, left, top = preprocess_yolo(frame)
        pred = self.sess.run([self.output_name], {self.input_name: img})[0]
        return postprocess_yolo(pred, scale, left, top, frame.shape)

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
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.transpose(arr, (2, 0, 1))
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
        db = np.vstack(self.features)
        sims = np.dot(db, emb)
        dists = 1.0 - sims
        idx = np.argmin(dists)
        return self.labels[idx], float(dists[idx])

if __name__ == "__main__":
    cap = cv2.VideoCapture(DEVICE_ID)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    # Set capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    proc = psutil.Process()
    buf_fps = collections.deque(maxlen=30)
    buf_cpu = collections.deque(maxlen=30)
    buf_mem = collections.deque(maxlen=30)
    buf_gpu = collections.deque(maxlen=30)
    prev = time.time()
    frame_count = 0

    detector = FaceDetectorYOLO()
    recognizer = FaceRecognizerArcFaceTorch(model_path="models/arcface.pt", threshold=0.3)

    training_mode = False
    target_samples = 250
    sample_count = 0
    current_label = None

    print("[T] Train  |  [Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # System metrics
        now = time.time()
        fps = 1.0 / (now - prev) if now > prev else 0.0
        prev = now
        buf_fps.append(fps)
        cpu = proc.cpu_percent() / psutil.cpu_count()
        buf_cpu.append(cpu)
        mem = proc.memory_info().rss / (1024 * 1024)
        buf_mem.append(mem)
        frame_count += 1
        if frame_count % 10 == 0:
            gpus = GPUtil.getGPUs()
            load = gpus[0].load * 100.0 if gpus else 0.0
            buf_gpu.append(load)

        avg_fps = sum(buf_fps) / len(buf_fps) if buf_fps else 0.0
        avg_cpu = sum(buf_cpu) / len(buf_cpu) if buf_cpu else 0.0
        avg_mem = sum(buf_mem) / len(buf_mem) if buf_mem else 0.0
        avg_gpu = sum(buf_gpu) / len(buf_gpu) if buf_gpu else 0.0

        faces = detector.detect(frame)

        if training_mode and faces:
            (x1, y1, x2, y2), _ = faces[0]
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue
            roi = frame[y1 : y2, x1 : x2]
            if roi.size == 0:
                continue
            recognizer.add_sample(roi, current_label)
            sample_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                "Train '{}': {}/{}".format(current_label, sample_count, target_samples),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            if sample_count >= target_samples:
                training_mode = False
                sample_count = 0
                print("Training completed for '{}'".format(current_label))
        else:
            for (x1, y1, x2, y2), score in faces:
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue
                roi = frame[y1 : y2, x1 : x2]
                if roi.size == 0:
                    continue
                label, dist = recognizer.recognize(roi)
                if dist < recognizer.threshold:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    "{} ({:.2f})".format(label, dist),
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

        # Overlay metrics
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
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("t"):
            lbl = input("Label: ").strip()
            if lbl:
                current_label = lbl
                training_mode = True

    cap.release()
    cv2.destroyAllWindows()
