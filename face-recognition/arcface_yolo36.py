import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from PIL import Image

# Configuración YOLO
MODEL_PATH = "models/yolov8n-face.onnx"
INPUT_SIZE = 640
CONF_THRES = 0.25
NMS_THRES = 0.45
DEVICE_ID = 0

# Funciones YOLO
from pathlib import Path

def letterbox(img, new_size=INPUT_SIZE, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = img_resized
    return canvas, scale, left, top


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def iou(box, boxes):
    inter = (np.maximum(0, np.minimum(boxes[:, 2], box[2]) - np.maximum(boxes[:, 0], box[0])) *
             np.maximum(0, np.minimum(boxes[:, 3], box[3]) - np.maximum(boxes[:, 1], box[1])))
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

# Detector usando YOLO ONNX
class FaceDetectorYOLO:
    def __init__(self, model_path=MODEL_PATH):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Modelo ONNX no encontrado en {model_path}")
        self.sess = ort.InferenceSession(model_path, providers=get_execution_providers())
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def detect(self, frame):
        img, scale, left, top = preprocess_yolo(frame)
        pred = self.sess.run([self.output_name], {self.input_name: img})[0]
        return postprocess_yolo(pred, scale, left, top, frame.shape)

# Reconocedor ArcFace (igual que antes)
class FaceRecognizerArcFaceTorch:
    def __init__(self, model_path="models/arcface.pt", recognition_threshold=0.3, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.to(self.device); self.model.eval()
        self.features_database = []
        self.labels = []
        self.recognition_threshold = recognition_threshold
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def preprocess(self, face_image):
        pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        return self.transform(pil).unsqueeze(0).to(self.device)

    def extract_features(self, face_image):
        tensor = self.preprocess(face_image)
        with torch.no_grad(): emb = self.model(tensor).flatten().cpu().numpy()
        norm = np.linalg.norm(emb)
        return emb / norm if norm>0 else emb

    def add_training_sample(self, face_image, label):
        emb = self.extract_features(face_image)
        self.features_database.append(emb); self.labels.append(label)

    def recognize(self, face_image):
        emb = self.extract_features(face_image)
        if not self.features_database:
            return "Unknown", 1.0
        sims = np.dot(self.features_database, emb)
        dists = 1 - sims
        idx = np.argmin(dists)
        return self.labels[idx], dists[idx]

# Ejecución principal sin persistencia
if __name__ == "__main__":
    cap = cv2.VideoCapture(DEVICE_ID)
    if not cap.isOpened(): raise RuntimeError("No se pudo abrir la cámara")

    detector = FaceDetectorYOLO()
    recognizer = FaceRecognizerArcFaceTorch()
    training_mode = False
    target_samples = 250
    sample_count = 0
    current_label = None

    print("[T] Entrenar nuevo label | [q] Salir")

    while True:
        ret, frame = cap.read()
        if not ret: break

        detections = detector.detect(frame)

        if training_mode and detections:
            (x1, y1, x2, y2), _ = detections[0]
            roi = frame[y1:y2, x1:x2]
            recognizer.add_training_sample(roi, current_label)
            sample_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Train {current_label}: {sample_count}/{target_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if sample_count >= target_samples:
                training_mode = False; sample_count = 0
                print(f"Entrenamiento completado para '{current_label}'")
        else:
            for (x1, y1, x2, y2), score in detections:
                roi = frame[y1:y2, x1:x2]
                label, dist = recognizer.recognize(roi)
                color = (0, 255, 0) if dist < recognizer.recognition_threshold else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({dist:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Feed YOLO", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('t'):
            lbl = input("Label: ").strip()
            if lbl: current_label = lbl; training_mode = True

    cap.release()
    cv2.destroyAllWindows()