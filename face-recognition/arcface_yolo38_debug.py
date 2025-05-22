import cv2
import numpy as np
import time
import collections
import psutil
import GPUtil
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image

MODEL_PATH = "models/yolov8n-face.pt"
DEVICE_ID = 0
TARGET_SAMPLES = 250

class FaceRecognizerArcFaceTorch:
    def __init__(self, path="models/arcface.pt", thresh=0.3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()
        self.features = []
        self.labels = []
        self.thresh = thresh
        self.tf = transforms.Compose([transforms.Resize((112,112)), transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
    def extract(self, roi):
        img = Image.fromarray(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
        t = self.tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad(): v=self.model(t).flatten().cpu().numpy()
        n=np.linalg.norm(v); return v/n if n>0 else v
    def add(self, roi, label): self.features.append(self.extract(roi)); self.labels.append(label)
    def recognize(self, roi):
        if not self.features: return "Unknown",1.0
        e=self.extract(roi); d=1-np.dot(self.features,e); i=np.argmin(d)
        return self.labels[i], d[i]

if __name__=="__main__":
    yolo = YOLO(MODEL_PATH).to('cuda' if torch.cuda.is_available() else 'cpu')
    cap = cv2.VideoCapture(DEVICE_ID)
    proc = psutil.Process()
    buf_fps = collections.deque(maxlen=30)
    buf_cpu = collections.deque(maxlen=30)
    buf_mem = collections.deque(maxlen=30)
    buf_gpu = collections.deque(maxlen=30)
    prev_time = time.time(); frame_count = 0

    rec = FaceRecognizerArcFaceTorch()
    mode=False; cnt=0; lbl=None

    print("[T] Train | [q] Quit")
    while True:
        ret, frame = cap.read()
        if not ret: break
        now = time.time(); fps = 1/(now-prev_time); prev_time = now
        buf_fps.append(fps)
        buf_cpu.append(proc.cpu_percent()/psutil.cpu_count())
        buf_mem.append(proc.memory_info().rss/1024/1024)
        frame_count+=1
        if frame_count%10==0:
            gpus = GPUtil.getGPUs(); load = gpus[0].load*100 if gpus else 0; buf_gpu.append(load)
        avg_fps = sum(buf_fps)/len(buf_fps)
        avg_cpu = sum(buf_cpu)/len(buf_cpu)
        avg_mem = sum(buf_mem)/len(buf_mem)
        avg_gpu = sum(buf_gpu)/len(buf_gpu) if buf_gpu else 0

        results = yolo(frame)
        boxes = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]-b.xyxy[0][0]), int(b.xyxy[0][3]-b.xyxy[0][1]))
                 for b in results[0].boxes]

        if mode and boxes:
            x,y,w,h=boxes[0]
            roi=frame[y:y+h,x:x+w]
            rec.add(roi,lbl); cnt+=1
            if cnt>=TARGET_SAMPLES: mode=False; cnt=0; print(f"Done '{lbl}'")
        else:
            for x,y,w,h in boxes:
                roi=frame[y:y+h,x:x+w]
                l,d = rec.recognize(roi)
                c = (0,255,0) if d<rec.thresh else (0,0,255)
                cv2.rectangle(frame,(x,y),(x+w,y+h),c,2)
                cv2.putText(frame,f"{l} ({d:.2f})",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,c,2)

        # Overlay métricas
        cv2.putText(frame,f"FPS:{int(avg_fps)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(100,255,0),2)
        cv2.putText(frame,f"CPU:{avg_cpu:.1f}%",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(100,255,0),2)
        cv2.putText(frame,f"Mem:{avg_mem:.1f}MB",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.8,(100,255,0),2)
        cv2.putText(frame,f"GPU:{avg_gpu:.1f}%",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.8,(100,255,0),2)

        cv2.imshow("Feed YOLO Metrics", frame)
        k=cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        if k==ord('t'):
            lbl = input("Label: ").strip(); mode = bool(lbl)
    cap.release(); cv2.destroyAllWindows()