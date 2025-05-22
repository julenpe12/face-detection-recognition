import cv2
import numpy as np
import time
import collections
import psutil
import GPUtil
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

# YOLO (idénticas funciones que arriba)
MODEL_PATH = "models/yolov8n-face.onnx"; INPUT_SIZE=640; CONF_THRES=0.25; NMS_THRES=0.45; DEVICE_ID=0

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

class FaceDetectorYOLO:
    def __init__(self, model_path=MODEL_PATH):
        if not Path(model_path).is_file(): raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.sess = ort.InferenceSession(model_path, providers=get_execution_providers())
        self.inp = self.sess.get_inputs()[0].name; self.outp = self.sess.get_outputs()[0].name
    def detect(self, frame):
        img, sc, l, t0 = preprocess_yolo(frame)
        pred = self.sess.run([self.outp], {self.inp: img})[0]
        return postprocess_yolo(pred, sc, l, t0, frame.shape)

class FaceRecognizerArcFaceTorch:
    def __init__(self, path="models/arcface.pt",thresh=0.3):
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model=torch.jit.load(path,map_location=self.device); self.model.eval()
        self.features=[]; self.labels=[]; self.thresh=thresh
        self.tf=transforms.Compose([transforms.Resize((112,112)),transforms.ToTensor(),transforms.Normalize([0.5]*3,[0.5]*3)])
    def prep(self,roi):
        img=Image.fromarray(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
        return self.tf(img).unsqueeze(0).to(self.device)
    def ext(self,roi):
        t=self.prep(roi)
        with torch.no_grad(): v=self.model(t).flatten().cpu().numpy()
        n=np.linalg.norm(v); return v/n if n>0 else v
    def add(self,roi,label): self.features.append(self.ext(roi)); self.labels.append(label)
    def recognize(self,roi):
        if not self.features: return "Unknown",1.0
        e=self.ext(roi);d=1-np.dot(self.features,e);i=np.argmin(d)
        return self.labels[i],d[i]

if __name__=="__main__":
    cap=cv2.VideoCapture(DEVICE_ID)
    proc=psutil.Process();
    buf_fps=collections.deque(maxlen=30); buf_cpu=collections.deque(maxlen=30);
    buf_mem=collections.deque(maxlen=30); buf_gpu=collections.deque(maxlen=30);
    prev=time.time(); count=0

    det=FaceDetectorYOLO(); rec=FaceRecognizerArcFaceTorch()
    mode=False; cnt=0; lbl=None; tgt=250

    print("[T] Train | [q] Quit")
    while True:
        ret,frame=cap.read();
        if not ret: break
        now=time.time(); fps=1/(now-prev); prev=now; buf_fps.append(fps)
        cpu=proc.cpu_percent()/psutil.cpu_count(); buf_cpu.append(cpu)
        mem=proc.memory_info().rss/1024/1024; buf_mem.append(mem)
        count+=1
        if count%10==0:
            gpus=GPUtil.getGPUs(); load=gpus[0].load*100 if gpus else 0; buf_gpu.append(load)
        avg_fps=sum(buf_fps)/len(buf_fps); avg_cpu=sum(buf_cpu)/len(buf_cpu)
        avg_mem=sum(buf_mem)/len(buf_mem); avg_gpu=sum(buf_gpu)/len(buf_gpu)

        faces=det.detect(frame)
        if mode and faces:
            (x1,y1,x2,y2),_ = faces[0]
            roi=frame[y1:y2,x1:x2]; rec.add(roi,lbl); cnt+=1
            if cnt>=tgt: mode=False; cnt=0; print(f"Done '{lbl}'")
        else:
            for (x1,y1,x2,y2),_ in faces:
                roi=frame[y1:y2,x1:x2]; l,d=rec.recognize(roi)
                c=(0,255,0) if d<rec.thresh else (0,0,255)
                cv2.rectangle(frame,(x1,y1),(x2,y2),c,2)
                cv2.putText(frame,f"{l} ({d:.2f})",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,c,2)

        cv2.putText(frame,f"FPS:{int(avg_fps)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(100,255,0),2)
        cv2.putText(frame,f"CPU:{avg_cpu:.1f}%",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(100,255,0),2)
        cv2.putText(frame,f"Mem:{avg_mem:.1f}MB",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.8,(100,255,0),2)
        cv2.putText(frame,f"GPU:{avg_gpu:.1f}%",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.8,(100,255,0),2)

        cv2.imshow("Feed YOLO Metrics", frame)
        k=cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        if k==ord('t'): lbl=input("Label: ").strip(); mode=bool(lbl)
    cap.release(); cv2.destroyAllWindows()