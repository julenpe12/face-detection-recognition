import cv2
import numpy as np
import pickle
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

MODEL_FILE = "trained/arcface.pkl"

class FaceDetectorHaar:
    def __init__(self, cascade_path=cv2.data.haarcascades+"haarcascade_frontalface_default.xml"):
        self.cascade = cv2.CascadeClassifier(cascade_path)
    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.cascade.detectMultiScale(gray,1.1,5)

class FaceRecognizerArcFaceTorch:
    def __init__(self, path="models/arcface.pt", thresh=0.3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.jit.load(path,map_location=self.device)
        self.model.eval()
        self.features=[]
        self.labels=[]
        self.thresh=thresh
        self.tf = transforms.Compose([transforms.Resize((112,112)),transforms.ToTensor(),transforms.Normalize([0.5]*3,[0.5]*3)])
    
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

    def preprocess(self,roi):
        img = Image.fromarray(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
        return self.tf(img).unsqueeze(0).to(self.device)
    def extract(self,roi):
        t = self.preprocess(roi)
        with torch.no_grad(): v = self.model(t).flatten().cpu().numpy()
        n = np.linalg.norm(v)
        return v/n if n>0 else v
    def add(self,roi,label):
        self.features.append(self.extract(roi)); self.labels.append(label)
    def recognize(self,roi):
        if not self.features: return "Unknown",1.0
        e = self.extract(roi)
        sims = np.dot(self.features,e)
        d = 1-sims
        i = np.argmin(d)
        return self.labels[i], d[i]

def save_model(r):
    os.makedirs(os.path.dirname(MODEL_FILE),exist_ok=True)
    with open(MODEL_FILE,'wb') as f: pickle.dump(r,f)

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE,'rb') as f: return pickle.load(f)
    return None

if __name__=="__main__":
    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): raise RuntimeError("No camera")
    det=FaceDetectorHaar()
    rec=load_model() or FaceRecognizerArcFaceTorch()
    mode=False; cnt=0; tgt=250; lbl=None
    print("[T] Train | [q] Quit")
    while True:
        ret,frame=cap.read()
        if not ret: break
        faces=det.detect(frame)
        if mode and len(faces)>0:
            x,y,w,h=faces[0]
            roi=frame[y:y+h,x:x+w]
            rec.add(roi,lbl); cnt+=1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,f"{cnt}/{tgt}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            if cnt>=tgt:
                mode=False; cnt=0
                save_model(rec)
                print(f"Saved label '{lbl}'")
        else:
            for x,y,w,h in faces:
                roi=frame[y:y+h,x:x+w]
                l,d=rec.recognize(roi)
                c=(0,255,0) if d<rec.thresh else (0,0,255)
                cv2.rectangle(frame,(x,y),(x+w,y+h),c,2)
                cv2.putText(frame,f"{l} ({d:.2f})",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,c,2)
        cv2.imshow("Feed",frame)
        k=cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        if k==ord('t'):
            lbl=input("Label: ").strip()
            if lbl: mode=True; print(f"Training '{lbl}'")
    cap.release(); cv2.destroyAllWindows()