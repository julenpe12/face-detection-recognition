import cv2
import numpy as np
import pickle
import os
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

# YOLO (idéntico a arriba)
MODEL_PATH = "models/yolov8n-face.onnx"; INPUT_SIZE = 640; CONF_THRES = 0.25; NMS_THRES = 0.45; DEVICE_ID = 0

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

class FaceDetectorYOLO:  # Mismo contenido que Script 1
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

MODEL_FILE = "trained/arcface.pkl"

class FaceRecognizerArcFaceTorch:  # Igual que antes
    def __init__(self, path="models/arcface.pt", thresh=0.3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()
        self.features = []
        self.labels = []
        self.thresh = thresh
        self.tf = transforms.Compose([transforms.Resize((112,112)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = None
        return state

    def __setstate__(self, state):
        # Manejar objetos pickle antiguos sin model_path
        if 'model_path' not in state:
            state['model_path'] = "models/arcface.pt"
        self.__dict__.update(state)
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.to(self.device); self.model.eval()
    def preprocess(self, roi):
        img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        return self.tf(img).unsqueeze(0).to(self.device)
    def extract(self, roi):
        t = self.preprocess(roi)
        with torch.no_grad(): v = self.model(t).flatten().cpu().numpy()
        n = np.linalg.norm(v)
        return v/n if n>0 else v
    def add(self, roi, label): self.features.append(self.extract(roi)); self.labels.append(label)
    def recognize(self, roi):
        if not self.features: return "Unknown",1.0
        e = self.extract(roi)
        sims = np.dot(self.features, e)
        d = 1 - sims
        i = np.argmin(d)
        return self.labels[i], d[i]


def save_model(r): os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True);
    with open(MODEL_FILE,'wb') as f: pickle.dump(r,f)

def load_model():
    return pickle.load(open(MODEL_FILE,'rb')) if os.path.exists(MODEL_FILE) else None

if __name__ == "__main__":
    cap = cv2.VideoCapture(DEVICE_ID)
    if not cap.isOpened(): raise RuntimeError("No camera")
    detector = FaceDetectorYOLO()
    recognizer = load_model() or FaceRecognizerArcFaceTorch()
    mode = False; cnt = 0; tgt = 250; lbl = None
    print("[T] Train | [q] Quit")
    while True:
        ret, frame = cap.read();
        if not ret: break
        faces = detector.detect(frame)
        if mode and faces:
            (x1,y1,x2,y2),_ = faces[0]
            roi = frame[y1:y2, x1:x2]
            recognizer.add(roi,lbl); cnt += 1
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(frame,f"{cnt}/{tgt}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            if cnt>=tgt:
                mode=False; cnt=0; save_model(recognizer)
                print(f"Saved '{lbl}'")
        else:
            for (x1,y1,x2,y2),score in faces:
                roi = frame[y1:y2, x1:x2]
                l,d = recognizer.recognize(roi)
                c = (0,255,0) if d<recognizer.thresh else (0,0,255)
                cv2.rectangle(frame,(x1,y1),(x2,y2),c,2)
                cv2.putText(frame,f"{l} ({d:.2f})",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,c,2)
        cv2.imshow("Feed YOLO",frame)
        k = cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        if k==ord('t'):
            lbl = input("Label: ").strip(); mode = bool(lbl)
    cap.release(); cv2.destroyAllWindows()