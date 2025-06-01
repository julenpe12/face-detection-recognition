########################################
# Script 3: Sin persistencia, con métricas de sistema
########################################
import cv2
import numpy as np
import time
import collections
import psutil
import GPUtil

MODEL_PATH = "models/arcface_int8.onnx"
RECOGNITION_THRESHOLD = 0.3

class FaceDetectorHaar:
    def __init__(self, cascade_path=cv2.data.haarcascades + "haarcascade_frontalface_default.xml"):
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.cascade.detectMultiScale(gray)

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
        resized = cv2.resize(rgb, (112,112))
        return cv2.dnn.blobFromImage(resized)

    def extract(self, img):
        blob = self.preprocess(img)
        self.model.setInput(blob)
        emb = self.model.forward().flatten()
        norm = np.linalg.norm(emb)
        return emb/norm if norm>0 else emb

    def add(self, img, label):
        self.features_database.append(self.extract(img))
        self.labels.append(label)

    def recognize(self, img):
        emb = self.extract(img)
        if not self.features_database:
            return "Unknown", 1.0
        sims = np.dot(self.features_database, emb)
        dists = 1 - sims
        i = np.argmin(dists)
        return self.labels[i], dists[i]


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    detector = FaceDetectorHaar()
    recognizer = FaceRecognizerArcFace()

    process = psutil.Process()
    cores = psutil.cpu_count(logical=True)

    fps_buf = collections.deque(maxlen=30)
    cpu_buf = collections.deque(maxlen=30)
    mem_buf = collections.deque(maxlen=30)
    gpu_buf = collections.deque(maxlen=30)
    inf_buf = collections.deque(maxlen=30)
    prev_time = time.time()

    training = False
    label = None
    cnt = 0
    target = 250

    print("'T' entrenar, 'q' salir.")
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Métricas
        now = time.time()
        fps_buf.append(1/(now-prev_time))
        prev_time = now
        cpu_buf.append(process.cpu_percent()/cores)
        mem_buf.append(process.memory_info().rss/1024/1024)
        if len(fps_buf)%10==0:
            g = GPUtil.getGPUs()
            gpu_buf.append(g[0].load*100 if g else 0)
        avg_fps = sum(fps_buf)/len(fps_buf)
        avg_cpu = sum(cpu_buf)/len(cpu_buf)
        avg_mem = sum(mem_buf)/len(mem_buf)
        avg_gpu = sum(gpu_buf)/len(gpu_buf) if gpu_buf else 0

        faces = detector.detect(frame)
        if training and len(faces)>0:
            x,y,w,h = faces[0]
            roi = frame[y:y+h, x:x+w]
            start = time.time()
            recognizer.add(roi, label)
            inf_buf.append((time.time()-start)*1000)
            cnt += 1
            if cnt>=target:
                training=False
                cnt=0
                print(f"Entrenado '{label}'.")

        else:
            for (x,y,w,h) in faces:
                roi = frame[y:y+h, x:x+w]
                start = time.time()
                lbl, d = recognizer.recognize(roi)
                inf_buf.append((time.time()-start)*1000)
                color = (0,255,0) if d<recognizer.threshold else (0,0,255)
                cv2.putText(frame, f"{lbl} ({d:.2f})", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(frame, (x,y),(x+w,y+h), color, 2)

        # Mostrar métricas
        cv2.putText(frame, f"FPS: {avg_fps:.0f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,0), 2)
        cv2.putText(frame, f"CPU%: {avg_cpu*100:.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,0), 2)
        cv2.putText(frame, f"RAM: {avg_mem:.1f}MB", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,0), 2)
        cv2.putText(frame, f"GPU%: {avg_gpu:.1f}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,0), 2)

        cv2.imshow("Feed Metrics", frame)
        k = cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        if k in (ord('T'),):
            label = input("Etiqueta: ").strip()
            if label:
                training=True
                cnt=0

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()