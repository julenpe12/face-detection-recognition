import cv2
import numpy as np
torch_import = True
import torch
import torchvision.transforms as transforms
from PIL import Image

# Detector Haar Cascade
class FaceDetectorHaar:
    def __init__(self, cascade_path=cv2.data.haarcascades + "haarcascade_frontalface_default.xml", scaleFactor=1.1, minNeighbors=5):
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors)
        return faces

# Reconocedor ArcFace
class FaceRecognizerArcFaceTorch:
    def __init__(self, model_path="models/arcface.pt", recognition_threshold=0.3, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.features_database = []
        self.labels = []
        self.recognition_threshold = recognition_threshold
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def preprocess(self, face_image):
        pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil).unsqueeze(0)
        return tensor.to(self.device)

    def extract_features(self, face_image):
        tensor = self.preprocess(face_image)
        with torch.no_grad():
            emb = self.model(tensor).flatten().cpu().numpy()
        norm = np.linalg.norm(emb)
        return emb / norm if norm>0 else emb

    def add_training_sample(self, face_image, label):
        emb = self.extract_features(face_image)
        self.features_database.append(emb)
        self.labels.append(label)

    def recognize(self, face_image):
        emb = self.extract_features(face_image)
        if not self.features_database:
            return "Unknown", 1.0
        sims = np.dot(self.features_database, emb)
        dists = 1 - sims
        idx = np.argmin(dists)
        return self.labels[idx], dists[idx]

# Ejecución principal
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    detector = FaceDetectorHaar()
    recognizer = FaceRecognizerArcFaceTorch()
    training_mode = False
    target_samples = 250
    sample_count = 0
    current_label = None

    print("[T] Entrenar nuevo label | [q] Salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        # Condición corregida:
        if training_mode and len(faces) > 0:
            x, y, w, h = faces[0]
            roi = frame[y:y+h, x:x+w]
            recognizer.add_training_sample(roi, current_label)
            sample_count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame,
                        f"Train {current_label}: {sample_count}/{target_samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if sample_count >= target_samples:
                training_mode = False
                sample_count = 0
                print(f"Entrenamiento completado para '{current_label}'")
        else:
            # Aquí no es necesario comprobar de nuevo; el for no iterará si faces está vacío
            for (x, y, w, h) in faces:
                roi = frame[y:y+h, x:x+w]
                label, dist = recognizer.recognize(roi)
                color = (0, 255, 0) if dist < recognizer.recognition_threshold else (0, 0, 255)
                text = f"{label} ({dist:.2f})"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
        if key==ord('t'):
            current_label = input("Label: ").strip()
            if current_label:
                training_mode = True
                print(f"Modo entrenamiento: '{current_label}'")

    cap.release()
    cv2.destroyAllWindows()