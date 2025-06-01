import cv2
import numpy as np

# Ruta al archivo Haar Cascade (debe descargarse y colocarse en el mismo directorio)
CASCADE_PATH = "models/haarcascade_frontalface_default.xml"

MODEL_PATH = "models/arcface_int8.onnx"
RECOGNITION_THRESHOLD = 0.3

class FaceDetectorHaar:
    def __init__(self, cascade_path=CASCADE_PATH, scaleFactor=1.1, minNeighbors=5):
        if not cv2.os.path.exists(cascade_path):
            raise ValueError(f"Cascade file not found: {cascade_path}")
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors
        )
        return faces

class FaceRecognizerArcFace:
    def __init__(self, model_path=MODEL_PATH, recognition_threshold=RECOGNITION_THRESHOLD):
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.features_database = []
        self.labels = []
        self.recognition_threshold = recognition_threshold

    def preprocess(self, face_image):
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (112, 112))
        return cv2.dnn.blobFromImage(face_resized)

    def extract_features(self, face_image):
        blob = self.preprocess(face_image)
        self.model.setInput(blob)
        embedding = self.model.forward().flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def add_training_sample(self, face_image, label):
        emb = self.extract_features(face_image)
        self.features_database.append(emb)
        self.labels.append(label)

    def recognize(self, face_image):
        emb = self.extract_features(face_image)
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

    training_mode = False
    target_samples = 250
    current_label = None
    count = 0

    print("Presione 'T' para iniciar entrenamiento, 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        if training_mode and len(faces) > 0:
            x, y, w, h = faces[0]
            if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                continue
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            recognizer.add_training_sample(roi, current_label)
            count += 1
            cv2.putText(
                frame,
                f"Training {current_label}: {count}/{target_samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if count >= target_samples:
                training_mode = False
                count = 0
                print(f"Entrenamiento completado para '{current_label}'.")
        else:
            if not recognizer.features_database:
                cv2.putText(
                    frame,
                    "No hay datos de entrenamiento",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            for (x, y, w, h) in faces:
                if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                    continue
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                lbl, dist = recognizer.recognize(roi)
                color = (0, 255, 0) if dist < recognizer.recognition_threshold else (0, 0, 255)
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

        cv2.imshow("Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in (ord('T'), ord('t')) and not training_mode:
            current_label = input("Etiqueta para entrenamiento: ").strip()
            if current_label:
                training_mode = True
                count = 0
                print(f"Modo entrenamiento: '{current_label}'")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
